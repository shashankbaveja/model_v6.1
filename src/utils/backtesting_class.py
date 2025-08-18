"""
Backtester (project-agnostic backtesting engine)

Overview
--------
Provides a stable, reusable engine to backtest entry/exit signals against OHLCV data.
The engine is decoupled from any model-training pipeline and consumes standardized
inputs only. It supports T+1 execution semantics, volatility-based TP/SL/HP targets,
optional exit-signal integration, and optional re-entry updates.

Standardized inputs
-------------------
Required (either DataFrame or path):
- Raw OHLCV (raw_data_path or raw_df)
  Columns: instrument_token, timestamp, open, high, low, close
  Optional: tradingsymbol
- Unified signals (unified_signals_path or unified_signals_df)
  Columns: instrument_token, timestamp, entry_flag (0/1), exit_flag (0/1), capital_multiplier (float)

Optional (either DataFrame/Index or path):
- Features (features_data_path or features_df) to provide volatility_ewma_30d.
  If absent, volatility_ewma_30d is computed from raw (EWMA(30) of 30d rolling std).
  Columns: instrument_token, timestamp, volatility_ewma_30d
- Token map (token_to_symbol_df) to add tradingsymbol.
  Columns: instrument_token, tradingsymbol
- Trading days calendar (trading_days_path or trading_days_index). If absent, a
  calendar is built from data dates plus +1 year of weekdays.
- output_dir to persist CSVs (trade_log.csv, debug_log.csv, backtest_report.csv).

Configuration (constructor arguments)
-------------------------------------
- lookahead_candles (int): holding-period horizon in trading sessions
- volatility_tp_multipler (float): TP multiplier on volatility_ewma_30d
- volatility_sl_multipler (float): SL multiplier on volatility_ewma_30d
- exit_signal_integration (bool): if True, exit_flag triggers EXIT_SIGNAL exits (T+1 open)
- allow_reentry_updates (bool): if True, when entry_flag re-fires and close > entry_price,
  TP/SL/HP are re-based to current close (no resizing)
- python_executable (str|None): accepted for interface parity; unused internally
- target_function (Callable|None): override to customize TP/SL/HP; default mirrors volatility logic
- capital_per_trade (float): base notional for position sizing

Execution semantics and logic
-----------------------------
- Entry: when entry_flag == 1 and no active position for the instrument
  - Preferred execution: next bar (T+1) open
  - Sizing uses the signal-bar close (not T+1 open), matching legacy behavior
  - If T+1 bar is missing, a pending record is logged with exit_reason = "enter_today"
- Exit priority (evaluated on each bar for active trades):
  1) TP (price >= tp_price)
  2) SL (price <= sl_price)
  3) HP (timestamp >= hp_exit_date; exit at close)
  4) EXIT_SIGNAL (if exit_signal_integration and exit_flag == 1; executes at T+1 open)
- Targets: computed by target_function with inputs {entry_price, volatility_ewma_30d,
  entry_date, trading_days}. Default implementation:
  tp = entry_price * (1 + vol * volatility_tp_multipler)
  sl = entry_price * (1 - vol * volatility_sl_multipler)
  hp_exit_date = entry_date + lookahead_candles (trading sessions)
- Re-entry updates: if allow_reentry_updates and entry_flag == 1 and close > entry_price,
  recompute TP/SL/HP with entry_price := close (no position resize)
- End-of-simulation: any open trades are marked as "exit_today" if the last bar has
  exit_flag == 1 and exit_signal_integration is True; otherwise "Active".

Outputs
-------
- trade_log_df (DataFrame): one row per trade with lifecycle fields
  instrument_token, tradingsymbol, signal_date, entry_date, entry_price, num_shares,
  tp_price, tp_thresh, sl_price, sl_thresh, hp_exit_date, capital_multiplier,
  entry_flag, exit_flag, exit_date, exit_price, exit_reason in {TP, SL, HP, EXIT_SIGNAL,
  enter_today, exit_today, Active}, pnl
- debug_df (DataFrame): per-row state including trade_active, current_tp, current_sl,
  hp_exit_date, entry_source
- performance_metrics (dict) computed on completed trades only:
  Total_Trades, Win_Rate_%, Total_PnL, Profit_Factor, Avg_Trade_PnL, Avg_Return_%,
  Avg_Holding_Period_Days, Max_Drawdown_PnL, Max_Drawdown_%, Best_Trade_PnL,
  Worst_Trade_PnL, Target_Hit_Count, Stop_Loss_Count, Holding_Period_Count,
  Exit_Signal_Count, Pending_Entry_Count, Pending_Exit_Count, Active_On_Close_Count

Usage
-----
Programmatic:
    from backtesting_class import Backtester
    bt = Backtester(
        raw_data_path='data/processed/test_raw.parquet',
        unified_signals_path='data/signals/unified_signals_v3.csv',
        features_data_path='data/processed/backtest_features_with_vol_v3.parquet',
        lookahead_candles=3,
        volatility_tp_multipler=4.0,
        volatility_sl_multipler=4.0,
        exit_signal_integration=True,
        allow_reentry_updates=True,
        capital_per_trade=5000,
        output_dir='reports',
    )
    results = bt.run()

Notes & assumptions
-------------------
- Unified signals must already encode the entry filter (e.g., via exit model) by setting
  entry_flag/capital_multiplier accordingly.
- Volatility feature is optional; if missing, it is computed from raw OHLCV.
- Holding period days are based on trading sessions for HP target but calendar days
  are used for metric reporting of holding_period_days.
"""

import os
import math
from typing import Callable, Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np


class Backtester:
    """
    Project-agnostic backtesting engine.

    Inputs are standardized and decoupled from any model-training pipeline.

    Required inputs (either DataFrame or path-based):
      - raw_data_path or raw_df: OHLCV with columns
          ['instrument_token', 'timestamp', 'open', 'high', 'low', 'close']
          optional: 'tradingsymbol'
      - unified_signals_path or unified_signals_df: with columns
          ['instrument_token', 'timestamp', 'entry_flag', 'exit_flag', 'capital_multiplier']

    Optional inputs:
      - features_data_path or features_df: if present and includes 'volatility_ewma_30d',
        it will be merged; otherwise volatility will be computed from raw prices
      - token_to_symbol_df: DataFrame with ['instrument_token', 'tradingsymbol']
      - trading_days_index or trading_days_path: custom trading session calendar
      - output_dir: if provided, saves CSVs for logs and report

    Config:
      - lookahead_candles: int
      - volatility_tp_multipler: float
      - volatility_sl_multipler: float
      - exit_signal_integration: bool
      - allow_reentry_updates: bool
      - python_executable: str | None (accepted for interface parity; unused internally)
      - target_function: Callable to compute tp/sl/hp given context
      - capital_per_trade: float
    """

    def __init__(
        self,
        *,
        raw_data_path: Optional[str] = None,
        raw_df: Optional[pd.DataFrame] = None,
        unified_signals_path: Optional[str] = None,
        unified_signals_df: Optional[pd.DataFrame] = None,
        features_data_path: Optional[str] = None,
        features_df: Optional[pd.DataFrame] = None,
        token_to_symbol_df: Optional[pd.DataFrame] = None,
        trading_days_path: Optional[str] = None,
        trading_days_index: Optional[pd.DatetimeIndex] = None,
        output_dir: Optional[str] = None,
        # Config
        lookahead_candles: int = 3,
        volatility_tp_multipler: float = 4.0,
        volatility_sl_multipler: float = 4.0,
        exit_signal_integration: bool = True,
        allow_reentry_updates: bool = True,
        python_executable: Optional[str] = None,
        target_function: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        capital_per_trade: float = 20000.0,
    ) -> None:
        self.raw_data_path = raw_data_path
        self.raw_df = raw_df
        self.unified_signals_path = unified_signals_path
        self.unified_signals_df = unified_signals_df
        self.features_data_path = features_data_path
        self.features_df = features_df
        self.token_to_symbol_df = token_to_symbol_df
        self.trading_days_path = trading_days_path
        self.trading_days_index = trading_days_index
        self.output_dir = output_dir

        self.lookahead_candles = lookahead_candles
        self.volatility_tp_multipler = volatility_tp_multipler
        self.volatility_sl_multipler = volatility_sl_multipler
        self.exit_signal_integration = exit_signal_integration
        self.allow_reentry_updates = allow_reentry_updates
        self.python_executable = python_executable
        self.capital_per_trade = capital_per_trade

        self.target_function = target_function or self._default_target_function

        # Validate required inputs presence (raw + signals)
        if self.raw_data_path is None and self.raw_df is None:
            raise ValueError("Either raw_data_path or raw_df must be provided")
        if self.unified_signals_path is None and self.unified_signals_df is None:
            raise ValueError("Either unified_signals_path or unified_signals_df must be provided")

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    # -------------------- Public API --------------------
    def run(self) -> Dict[str, Any]:
        base_df = self._prepare_data()
        trading_days = self._ensure_trading_days(base_df)
        trade_log, debug_df = self._simulate(base_df, trading_days)
        trade_log_df = pd.DataFrame(trade_log)
        performance_metrics = self._calculate_performance_metrics(trade_log_df)

        if self.output_dir:
            self._persist_outputs(trade_log_df, debug_df, performance_metrics)

        return {
            "trade_log_df": trade_log_df,
            "debug_df": debug_df,
            "performance_metrics": performance_metrics,
        }

    # -------------------- Preparation --------------------
    def _prepare_data(self) -> pd.DataFrame:
        raw_df = self._load_any(self.raw_data_path, self.raw_df)
        raw_df = raw_df.copy()

        # Basic schema enforcement
        required_raw = ["instrument_token", "timestamp", "open", "high", "low", "close"]
        self._ensure_columns(raw_df, required_raw, context="raw OHLCV")
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
        raw_df = raw_df.dropna(subset=["timestamp"])  # ensure valid timestamps

        # Merge token to symbol if provided and not already present
        if (self.token_to_symbol_df is not None) and ("tradingsymbol" not in raw_df.columns):
            token_map = self.token_to_symbol_df[["instrument_token", "tradingsymbol"]].drop_duplicates()
            raw_df = pd.merge(raw_df, token_map, on="instrument_token", how="left")

        # Load and merge unified signals
        sig_df = self._load_any(self.unified_signals_path, self.unified_signals_df)
        required_signals = ["instrument_token", "tradingsymbol", "timestamp", "entry_flag", "exit_flag", "capital_multiplier"]
        self._ensure_columns(sig_df, required_signals, context="unified signals")
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"], errors="coerce")
        sig_df = sig_df.dropna(subset=["timestamp"]).copy()

        # Normalize types
        for col in ["entry_flag", "exit_flag", "capital_multiplier"]:
            if col in sig_df.columns:
                sig_df[col] = sig_df[col].fillna(0)

        # Merge signals into raw with left join so we keep the full OHLCV history
        base_df = pd.merge(
            raw_df,
            sig_df[["instrument_token", "tradingsymbol", "timestamp", "entry_flag", "exit_flag", "capital_multiplier"]],
            on=["instrument_token", "timestamp"],
            how="left",
        )
        base_df["entry_flag"] = base_df["entry_flag"].fillna(0).astype(int)
        base_df["exit_flag"] = base_df["exit_flag"].fillna(0).astype(int)
        base_df["capital_multiplier"] = base_df["capital_multiplier"].fillna(0).astype(float)

        # Attach features if present (to acquire volatility_ewma_30d)
        features_df = self._load_any(self.features_data_path, self.features_df)
        if features_df is not None and "volatility_ewma_30d" in features_df.columns:
            features_df = features_df[["instrument_token", "timestamp", "volatility_ewma_30d"]].copy()
            features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], errors="coerce")
            features_df = features_df.dropna(subset=["timestamp"])  # valid timestamps only
            base_df = pd.merge(
                base_df,
                features_df,
                on=["instrument_token", "timestamp"],
                how="left",
                suffixes=("", "_feat"),
            )

        # Compute volatility if missing
        if "volatility_ewma_30d" not in base_df.columns:
            base_df = self._compute_volatility_ewma(base_df)
        else:
            # If some rows still null, compute and fill only missing
            computed = self._compute_volatility_ewma(base_df)
            base_df["volatility_ewma_30d"] = base_df["volatility_ewma_30d"].fillna(computed["volatility_ewma_30d"])  # type: ignore

        # Sort and cleanup
        base_df = base_df.sort_values(by=["instrument_token", "timestamp"]).reset_index(drop=True)
        return base_df

    def _load_any(self, path: Optional[str], df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is not None:
            return df
        if path is None:
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
        if path.lower().endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _ensure_columns(self, df: pd.DataFrame, cols: list, *, context: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {context}: {missing}")

    def _compute_volatility_ewma(self, df: pd.DataFrame) -> pd.DataFrame:
        # Following the existing logic in trade_generator_v3: rolling std(30) then EWMA(span=30)
        df = df.copy()
        df = df.sort_values(["instrument_token", "timestamp"])  # ensure order
        df["daily_return"] = df.groupby("instrument_token")["close"].pct_change()
        rolling_std = df.groupby("instrument_token")["daily_return"].transform(lambda x: x.rolling(window=30).std())
        df["volatility_ewma_30d"] = rolling_std.ewm(span=30, adjust=False).mean()
        df.drop(columns=["daily_return"], inplace=True)
        return df

    def _ensure_trading_days(self, base_df: pd.DataFrame) -> pd.DatetimeIndex:
        if self.trading_days_index is not None:
            return self.trading_days_index
        if self.trading_days_path is not None:
            idx_df = self._load_any(self.trading_days_path, None)
            if idx_df is None:
                raise ValueError("trading_days_path provided but failed to load")
            # Try to locate a date column
            date_col = "date" if "date" in idx_df.columns else idx_df.columns[0]
            days = pd.to_datetime(idx_df[date_col]).dt.normalize().dropna().drop_duplicates().sort_values()
            return pd.DatetimeIndex(days)

        # Build from data + 1-year weekdays
        dates = pd.to_datetime(base_df["timestamp"]).dt.normalize().dropna().drop_duplicates().sort_values()
        last_known = dates.max() if not dates.empty else pd.Timestamp.today().normalize()
        start_future = max(pd.Timestamp.today().normalize(), last_known)
        end_future = start_future + pd.DateOffset(years=1)
        future_weekdays = pd.date_range(start=start_future, end=end_future, freq="B")
        combined = pd.DatetimeIndex(dates).union(future_weekdays).unique().sort_values()
        return combined

    # -------------------- Simulation --------------------
    def _get_next_candle_map(self, df: pd.DataFrame) -> Dict[Tuple[int, pd.Timestamp], Dict[str, Any]]:
        df = df.sort_values(by=["instrument_token", "timestamp"]).copy()
        next_candle_data = df.groupby("instrument_token").shift(-1)
        next_candle_map: Dict[Tuple[int, pd.Timestamp], Dict[str, Any]] = {}
        keys = [tuple(x) for x in df[["instrument_token", "timestamp"]].to_numpy()]
        records = next_candle_data.to_dict("records")
        for i, key in enumerate(keys):
            next_candle_map[key] = records[i]
        return next_candle_map

    def _add_trading_days(self, base_ts: pd.Timestamp, num_days: int, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
        if trading_days is None or len(trading_days) == 0:
            return pd.to_datetime(base_ts).normalize()
        base_date = pd.to_datetime(base_ts).normalize()
        arr = trading_days.values
        idx = np.searchsorted(arr, base_date.to_datetime64(), side="left")
        target_idx = idx + int(num_days)
        if target_idx >= len(arr):
            target_idx = len(arr) - 1
        return pd.Timestamp(trading_days[target_idx])

    def _default_target_function(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        entry_price: float = float(ctx["entry_price"])  # price basis
        vol: float = float(ctx["volatility_ewma_30d"]) if not pd.isna(ctx["volatility_ewma_30d"]) else 0.0
        tp_mult: float = float(self.volatility_tp_multipler)
        sl_mult: float = float(self.volatility_sl_multipler)
        lookahead: int = int(self.lookahead_candles)
        trading_days: pd.DatetimeIndex = ctx["trading_days"]
        entry_date: pd.Timestamp = ctx["entry_date"]

        tp_price = entry_price * (1 + vol * tp_mult)
        sl_price = entry_price * (1 - vol * sl_mult)
        hp_exit_date = self._add_trading_days(entry_date, lookahead, trading_days)
        return {"tp_price": tp_price, "sl_price": sl_price, "hp_exit_date": hp_exit_date}

    def _simulate(
        self,
        df: pd.DataFrame,
        trading_days: pd.DatetimeIndex,
    ) -> Tuple[list, Optional[pd.DataFrame]]:
        trade_log: list = []
        active_trades: Dict[int, Dict[str, Any]] = {}
        next_candle_map = self._get_next_candle_map(df)

        debug_df = df.copy()
        debug_df["trade_active"] = 0
        debug_df["current_tp"] = np.nan
        debug_df["current_sl"] = np.nan
        debug_df["hp_exit_date"] = pd.NaT
        debug_df["entry_source"] = ""

        for instrument, group in df.groupby("instrument_token"):
            for row in group.itertuples(index=True):
                idx = row.Index
                # Maintain debug state
                if instrument in active_trades:
                    debug_df.at[idx, "trade_active"] = 1
                    debug_df.at[idx, "current_tp"] = active_trades[instrument]["tp_price"]
                    debug_df.at[idx, "current_sl"] = active_trades[instrument]["sl_price"]
                    debug_df.at[idx, "hp_exit_date"] = active_trades[instrument]["hp_exit_date"]

                # Exit evaluation if trade is active
                if instrument in active_trades:
                    trade = active_trades[instrument]
                    exit_reason = None
                    exit_price = None

                    if row.high >= trade["tp_price"]:
                        exit_reason = "TP"
                        exit_price = trade["tp_price"]
                    elif row.low <= trade["sl_price"]:
                        exit_reason = "SL"
                        exit_price = trade["sl_price"]
                    elif row.timestamp >= trade["hp_exit_date"]:
                        exit_reason = "HP"
                        exit_price = row.close
                    elif self.exit_signal_integration and int(row.exit_flag) == 1:
                        nxt = next_candle_map.get((instrument, row.timestamp))
                        if nxt and not pd.isna(nxt.get("open", np.nan)):
                            exit_reason = "EXIT_SIGNAL"
                            exit_price = nxt["open"]

                    if exit_reason is not None:
                        trade["exit_date"] = row.timestamp
                        trade["exit_price"] = float(exit_price)
                        trade["exit_reason"] = exit_reason
                        trade["pnl"] = (trade["exit_price"] - trade["entry_price"]) * trade["num_shares"]
                        trade_log.append(trade)
                        del active_trades[instrument]
                        continue

                    # Re-entry updates without resizing
                    if self.allow_reentry_updates and int(row.entry_flag) == 1 and (row.close > trade["entry_price"]):
                        ctx = {
                            "entry_price": float(row.close),
                            "volatility_ewma_30d": float(row.volatility_ewma_30d) if not pd.isna(row.volatility_ewma_30d) else 0.0,
                            "entry_date": row.timestamp,
                            "trading_days": trading_days,
                        }
                        targets = self.target_function(ctx)
                        trade["tp_price"] = targets["tp_price"]
                        trade["sl_price"] = targets["sl_price"]
                        trade["hp_exit_date"] = targets["hp_exit_date"]
                        # Keep latest flags on trade for traceability
                        trade["entry_flag"] = int(row.entry_flag)
                        trade["exit_flag"] = int(row.exit_flag)

                else:
                    # No active trade: consider new entry when entry_flag is 1
                    if int(row.entry_flag) == 1:
                        # Capital multiplier comes from unified signals (per your spec)
                        capital_multiplier = float(row.capital_multiplier) if not pd.isna(row.capital_multiplier) else 0.0
                        if capital_multiplier <= 0:
                            # No capital allocated → skip entry
                            continue

                        nxt = next_candle_map.get((instrument, row.timestamp))
                        signal_date = row.timestamp

                        if nxt and not pd.isna(nxt.get("open", np.nan)):
                            # Execute at T+1 open, but size on signal-bar close to mirror original logic
                            entry_price = float(nxt["open"])  # execution price
                            entry_date = pd.to_datetime(nxt["timestamp"]) if "timestamp" in nxt else signal_date

                            ctx = {
                                "entry_price": entry_price,
                                "volatility_ewma_30d": float(row.volatility_ewma_30d) if not pd.isna(row.volatility_ewma_30d) else 0.0,
                                "entry_date": entry_date,
                                "trading_days": trading_days,
                            }
                            targets = self.target_function(ctx)

                            allocated_capital = float(self.capital_per_trade) * capital_multiplier
                            sizing_price = float(row.close) if row.close > 0 else entry_price
                            num_shares = math.floor(allocated_capital / sizing_price) if sizing_price > 0 else 0
                            if num_shares > 0:
                                active_trades[instrument] = {
                                    "instrument_token": instrument,
                                    "tradingsymbol": getattr(row, "tradingsymbol", None),
                                    "signal_date": signal_date,
                                    "entry_date": entry_date,
                                    "entry_price": entry_price,
                                    "num_shares": num_shares,
                                    "tp_price": targets["tp_price"],
                                    "tp_thresh": float(row.volatility_ewma_30d) * float(self.volatility_tp_multipler) if not pd.isna(row.volatility_ewma_30d) else np.nan,
                                    "sl_price": targets["sl_price"],
                                    "sl_thresh": float(row.volatility_ewma_30d) * float(self.volatility_sl_multipler) if not pd.isna(row.volatility_ewma_30d) else np.nan,
                                    "hp_exit_date": targets["hp_exit_date"],
                                    "capital_multiplier": capital_multiplier,
                                    "entry_flag": int(row.entry_flag),
                                    "exit_flag": int(row.exit_flag),
                                    "entry_source": "flag",
                                }
                        else:
                            # No T+1 candle → record pending entry for visibility
                            entry_price = float(row.close)
                            ctx = {
                                "entry_price": entry_price,
                                "volatility_ewma_30d": float(row.volatility_ewma_30d) if not pd.isna(row.volatility_ewma_30d) else 0.0,
                                "entry_date": signal_date,
                                "trading_days": trading_days,
                            }
                            targets = self.target_function(ctx)
                            trade_log.append({
                                "instrument_token": instrument,
                                "tradingsymbol": getattr(row, "tradingsymbol", None),
                                "signal_date": signal_date,
                                "entry_date": pd.NaT,
                                "entry_price": entry_price,
                                "num_shares": math.floor(float(self.capital_per_trade) * capital_multiplier / entry_price) if entry_price > 0 else 0,
                                "tp_price": targets["tp_price"],
                                "tp_thresh": float(row.volatility_ewma_30d) * float(self.volatility_tp_multipler) if not pd.isna(row.volatility_ewma_30d) else np.nan,
                                "sl_price": targets["sl_price"],
                                "sl_thresh": float(row.volatility_ewma_30d) * float(self.volatility_sl_multipler) if not pd.isna(row.volatility_ewma_30d) else np.nan,
                                "exit_date": pd.NaT,
                                "exit_price": np.nan,
                                "exit_reason": "enter_today",
                                "pnl": np.nan,
                                "capital_multiplier": capital_multiplier,
                                "entry_flag": int(row.entry_flag),
                                "exit_flag": int(row.exit_flag),
                                "entry_source": "flag",
                            })

        # Wrap up any remaining active trades at end
        last_rows_map = {instrument: group.iloc[-1] for instrument, group in df.groupby("instrument_token")}
        for instrument, trade in active_trades.items():
            last_row = last_rows_map.get(instrument)
            exit_reason = "Active"
            if last_row is not None and self.exit_signal_integration and int(last_row["exit_flag"]) == 1:
                exit_reason = "exit_today"
            trade["exit_date"] = pd.NaT
            trade["exit_price"] = np.nan
            trade["exit_reason"] = exit_reason
            trade["pnl"] = np.nan
            trade_log.append(trade)

        return trade_log, debug_df

    # -------------------- Metrics & Reporting --------------------
    def _calculate_performance_metrics(self, trade_log_df: pd.DataFrame, initial_capital: Optional[float] = None) -> Dict[str, Any]:
        if trade_log_df is None or trade_log_df.empty:
            return {
                "Total_Trades": 0, "Win_Rate_%": 0, "Total_PnL": 0, "Profit_Factor": 0,
                "Avg_Trade_PnL": 0, "Avg_Return_%": 0, "Max_Drawdown_PnL": 0, "Max_Drawdown_%": 0,
                "Best_Trade_PnL": 0, "Worst_Trade_PnL": 0, "Target_Hit_Count": 0, "Stop_Loss_Count": 0,
                "Holding_Period_Count": 0, "Exit_Signal_Count": 0,
                "Pending_Entry_Count": 0, "Pending_Exit_Count": 0, "Active_On_Close_Count": 0,
                "Avg_Holding_Period_Days": 0,
            }

        unfiltered = trade_log_df.copy()
        pending_entry_count = (unfiltered["exit_reason"] == "enter_today").sum()
        pending_exit_count = (unfiltered["exit_reason"] == "exit_today").sum()
        active_on_close_count = (unfiltered["exit_reason"] == "Active").sum()

        completed = unfiltered[~unfiltered["exit_reason"].isin(["enter_today", "exit_today", "Active"])].copy()
        if completed.empty:
            return {
                "Total_Trades": 0, "Win_Rate_%": 0, "Total_PnL": 0, "Profit_Factor": 0,
                "Avg_Trade_PnL": 0, "Avg_Return_%": 0, "Max_Drawdown_PnL": 0, "Max_Drawdown_%": 0,
                "Best_Trade_PnL": 0, "Worst_Trade_PnL": 0, "Target_Hit_Count": 0, "Stop_Loss_Count": 0,
                "Holding_Period_Count": 0, "Exit_Signal_Count": 0,
                "Pending_Entry_Count": pending_entry_count, "Pending_Exit_Count": pending_exit_count,
                "Active_On_Close_Count": active_on_close_count,
                "Avg_Holding_Period_Days": 0,
            }

        completed["entry_date"] = pd.to_datetime(completed["entry_date"])  # may contain NaT
        completed["exit_date"] = pd.to_datetime(completed["exit_date"])  # may contain NaT

        # EXIT_SIGNAL executes T+1 open; adjust effective exit date by +1 day for HP calculations
        exit_signal_mask = completed["exit_reason"] == "EXIT_SIGNAL"
        completed["effective_exit_date"] = completed["exit_date"] + pd.to_timedelta(exit_signal_mask.astype(int), unit="D")
        completed["holding_period_days"] = (
            completed["effective_exit_date"].dt.normalize() - completed["entry_date"].dt.normalize()
        ).dt.days

        completed["pnl_pct"] = (completed["pnl"] / completed["entry_price"]) * 100
        total_trades = len(completed)
        wins = completed[completed["pnl"] > 0]
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = float(completed["pnl"].sum())

        gross_profit = float(completed[completed["pnl"] > 0]["pnl"].sum())
        gross_loss = abs(float(completed[completed["pnl"] < 0]["pnl"].sum()))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

        # Equity curve by exit date
        pnl_by_date = completed.groupby(completed["exit_date"].dt.date)["pnl"].sum()
        base_capital = float(initial_capital) if initial_capital is not None else float(self.capital_per_trade)
        equity_curve = pnl_by_date.cumsum() + base_capital
        peak = equity_curve.cummax()
        drawdown = peak - equity_curve
        max_drawdown_pnl = float(drawdown.max()) if len(drawdown) > 0 else 0.0
        max_drawdown_pct = (max_drawdown_pnl / float(peak[drawdown.idxmax()])) * 100 if len(peak) > 0 and drawdown.max() > 0 else 0.0

        return {
            "Total_Trades": total_trades,
            "Win_Rate_%": win_rate,
            "Total_PnL": total_pnl,
            "Profit_Factor": profit_factor,
            "Avg_Trade_PnL": float(completed["pnl"].mean()),
            "Avg_Return_%": float(completed["pnl_pct"].mean()),
            "Avg_Holding_Period_Days": float(completed["holding_period_days"].mean()),
            "Max_Drawdown_PnL": max_drawdown_pnl,
            "Max_Drawdown_%": max_drawdown_pct,
            "Best_Trade_PnL": float(completed["pnl"].max()),
            "Worst_Trade_PnL": float(completed["pnl"].min()),
            "Target_Hit_Count": int((completed["exit_reason"] == "TP").sum()),
            "Stop_Loss_Count": int((completed["exit_reason"] == "SL").sum()),
            "Holding_Period_Count": int((completed["exit_reason"] == "HP").sum()),
            "Exit_Signal_Count": int((completed["exit_reason"] == "EXIT_SIGNAL").sum()),
            "Pending_Entry_Count": int(pending_entry_count),
            "Pending_Exit_Count": int(pending_exit_count),
            "Active_On_Close_Count": int(active_on_close_count),
        }

    def _persist_outputs(self, trade_log_df: pd.DataFrame, debug_df: Optional[pd.DataFrame], performance_metrics: Dict[str, Any]) -> None:
        if not self.output_dir:
            return
        trade_log_path = os.path.join(self.output_dir, "trade_log.csv")
        trade_log_df.to_csv(trade_log_path, index=False)
        if debug_df is not None:
            debug_log_path = os.path.join(self.output_dir, "debug_log.csv")
            debug_df.to_csv(debug_log_path, index=False)
        report_path = os.path.join(self.output_dir, "backtest_report.csv")
        pd.DataFrame([performance_metrics]).to_csv(report_path, index=False)


__all__ = ["Backtester"]

