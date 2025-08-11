import os
import sys
import glob
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_backfill.myKiteLib import system_initialization


def load_config(config_path='config/parameters.yml') -> dict:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def prepare_data_v3(config: dict) -> pd.DataFrame:
    print("--- Phase 1 (v3): Data Preparation ---")

    backtest_config = config.get('backtest_v3', {})
    python_executable = backtest_config.get('python_executable', config.get('python_executable', sys.executable))

    # Signal generation (reuse v2 generator)
    print("Generating unified entry/exit signals (v3 via v2 generator)...")
    entry_model_path_1 = backtest_config['entry_models'][0]['model_path']
    entry_model_path_2 = backtest_config['entry_models'][1]['model_path']
    exit_model_path = backtest_config['exit_model']['model_path']

    subprocess.run([
        python_executable, 'src/signal_generator_v2.py',
        '--entry-model-path-1', entry_model_path_1,
        '--entry-model-path-2', entry_model_path_2,
        '--exit-model-path', exit_model_path,
    ], check=True)

    # Load unified signals v2/v3
    signals_dir = 'data/signals'
    unified_signals_path = os.path.join(signals_dir, 'unified_signals_v2.csv')
    print(f"Loading unified signals for v3 from: {unified_signals_path}")
    master_df = pd.read_csv(unified_signals_path)
    master_df['timestamp'] = pd.to_datetime(master_df['timestamp'])

    # Volatility
    print("Calculating volatility (EWMA 30d)...")
    master_df.sort_values(by=['instrument_token', 'timestamp'], inplace=True)
    master_df['daily_return'] = master_df.groupby('instrument_token')['close'].pct_change()
    rolling_std = master_df.groupby('instrument_token')['daily_return'].transform(lambda x: x.rolling(window=30).std())
    master_df['volatility_ewma_30d'] = rolling_std.ewm(span=30, adjust=False).mean()
    master_df.dropna(subset=['volatility_ewma_30d'], inplace=True)

    # Token to symbol mapping
    systemDetails = system_initialization()
    token_name_list = systemDetails.run_query_full("Select distinct instrument_token, tradingsymbol from kiteconnect.instruments_zerodha")
    token_name = pd.DataFrame(token_name_list)
    master_df = pd.merge(master_df, token_name, on='instrument_token', how='left')

    print("Data preparation (v3) complete.")
    return master_df


def get_next_candle_map(df: pd.DataFrame) -> dict:
    print("Pre-computing next candle data map (v3)...")
    df = df.sort_values(by=['instrument_token', 'timestamp']).copy()
    next_candle_data = df.groupby('instrument_token').shift(-1)
    next_candle_map = {}
    keys = [tuple(x) for x in df[['instrument_token', 'timestamp']].to_numpy()]
    records = next_candle_data.to_dict('records')
    for i, key in enumerate(keys):
        next_candle_map[key] = records[i]
    return next_candle_map


def run_simulation_v3(
    df: pd.DataFrame,
    config: dict,
    precomputed_next_candle_map: dict | None = None,
    precomputed_last_rows_map: dict | None = None,
    produce_debug: bool = True,
):
    print("\n--- Phase 2 (v3): Backtesting Simulation ---")
    backtest_config = config['backtest_v3']
    target_generation = config['target_generation']

    # Fixed thresholds from config
    exit_entry_filter_threshold = backtest_config['exit_model']['threshold_for_entry_filter']
    exit_trigger_threshold = backtest_config['exit_model']['threshold_for_exit_signal']

    trade_log = []
    active_trades = {}
    next_candle_map = precomputed_next_candle_map if precomputed_next_candle_map is not None else get_next_candle_map(df)

    debug_df = None
    if produce_debug:
        debug_df = df.copy()
        debug_df['trade_active'] = 0
        debug_df['current_tp'] = np.nan
        debug_df['current_sl'] = np.nan
        debug_df['hp_exit_date'] = pd.NaT
        debug_df['entry_source'] = ''

    for instrument, group in df.groupby('instrument_token'):
        for row in group.itertuples(index=True):
            idx = row.Index
            if instrument in active_trades:
                trade = active_trades[instrument]
                if produce_debug:
                    debug_df.loc[idx, 'trade_active'] = 1
                    debug_df.loc[idx, 'current_tp'] = trade['tp_price']
                    debug_df.loc[idx, 'current_sl'] = trade['sl_price']
                    debug_df.loc[idx, 'hp_exit_date'] = trade['hp_exit_date']

                exit_reason = None
                exit_price = None

                # TP/SL/HP priority
                if row.high >= trade['tp_price']:
                    exit_reason = 'TP'
                    exit_price = trade['tp_price']
                elif row.low <= trade['sl_price']:
                    exit_reason = 'SL'
                    exit_price = trade['sl_price']
                elif row.timestamp >= trade['hp_exit_date']:
                    exit_reason = 'HP'
                    exit_price = row.close
                elif backtest_config.get('exit_signal_integration', True) and row.exit_signal_prob > exit_trigger_threshold:
                    nxt = next_candle_map.get((instrument, row.timestamp))
                    if nxt and not pd.isna(nxt['open']):
                        exit_reason = 'EXIT_SIGNAL'
                        exit_price = nxt['open']

                if exit_reason:
                    trade['exit_date'] = row.timestamp
                    trade['exit_price'] = exit_price
                    trade['exit_reason'] = exit_reason
                    trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['num_shares']
                    trade_log.append(trade)
                    del active_trades[instrument]
                    continue

                # Re-entry updates (no re-sizing)
                if backtest_config.get('allow_reentry_updates', True):
                    # Evaluate tier flags again
                    m6 = (row.entry_signal_prob_1 >= 0.78 and row.entry_signal_prob_2 >= 0.77)
                    m4 = (row.entry_signal_prob_1 >= 0.70 and row.entry_signal_prob_2 >= 0.70)
                    m1 = (row.entry_signal_prob_1 >= 0.65 and row.entry_signal_prob_2 >= 0.65)
                    exit_filter_ok = row.exit_signal_prob < exit_entry_filter_threshold
                    entry_ok = (m6 or m4 or m1) and exit_filter_ok
                    if entry_ok and (row.close > trade['entry_price']):
                        new_base = row.close
                        trade['tp_price'] = new_base * (1 + row.volatility_ewma_30d * target_generation['volatility_tp_multipler'])
                        trade['sl_price'] = new_base * (1 - row.volatility_ewma_30d * target_generation['volatility_sl_multipler'])
                        trade['hp_exit_date'] = pd.to_datetime(row.timestamp.date() + timedelta(days=target_generation['lookahead_candles']))
                        trade['entry_signal_prob_1'] = row.entry_signal_prob_1
                        trade['entry_signal_prob_2'] = row.entry_signal_prob_2
                        trade['exit_signal_prob'] = row.exit_signal_prob

            else:
                # No active trade: three-tier entry + exit filter
                m6 = (row.entry_signal_prob_1 >= 0.78 and row.entry_signal_prob_2 >= 0.77)
                m4 = (row.entry_signal_prob_1 >= 0.70 and row.entry_signal_prob_2 >= 0.70)
                m1 = (row.entry_signal_prob_1 >= 0.65 and row.entry_signal_prob_2 >= 0.65)
                capital_multiplier = 0
                if m6:
                    capital_multiplier = 6
                elif m4:
                    capital_multiplier = 4
                elif m1:
                    capital_multiplier = 1

                exit_filter_ok = row.exit_signal_prob < exit_entry_filter_threshold
                if capital_multiplier > 0 and exit_filter_ok:
                    nxt = next_candle_map.get((instrument, row.timestamp))
                    signal_date = row.timestamp

                    tp_thresh = row.volatility_ewma_30d * target_generation['volatility_tp_multipler']
                    sl_thresh = row.volatility_ewma_30d * target_generation['volatility_sl_multipler']

                    if nxt and not pd.isna(nxt['open']):
                        entry_price = nxt['open']
                        entry_date = nxt['timestamp']
                        base_capital = backtest_config.get('capital_per_trade', 20000)
                        trade_capital = base_capital * capital_multiplier
                        num_shares = np.floor(trade_capital / row.close) if row.close > 0 else 0
                        if num_shares > 0:
                            tp_price = entry_price * (1 + row.volatility_ewma_30d * target_generation['volatility_tp_multipler'])
                            sl_price = entry_price * (1 - row.volatility_ewma_30d * target_generation['volatility_sl_multipler'])
                            hp_exit_date = pd.to_datetime(entry_date.date() + timedelta(days=target_generation['lookahead_candles']))
                            active_trades[instrument] = {
                                'instrument_token': instrument,
                                'tradingsymbol': row.tradingsymbol,
                                'signal_date': signal_date,
                                'entry_date': entry_date,
                                'entry_price': entry_price,
                                'num_shares': num_shares,
                                'tp_price': tp_price,
                                'tp_thresh': tp_thresh,
                                'sl_price': sl_price,
                                'sl_thresh': sl_thresh,
                                'hp_exit_date': hp_exit_date,
                                'capital_multiplier': capital_multiplier,
                                'entry_signal_prob_1': row.entry_signal_prob_1,
                                'entry_signal_prob_2': row.entry_signal_prob_2,
                                'exit_signal_prob': row.exit_signal_prob,
                                'entry_source': 'tiered',
                            }
                    else:
                        # No T+1 candle
                        entry_price = row.close
                        tp_price = entry_price * (1 + row.volatility_ewma_30d * target_generation['volatility_tp_multipler'])
                        sl_price = entry_price * (1 - row.volatility_ewma_30d * target_generation['volatility_sl_multipler'])
                        trade_log.append({
                            'instrument_token': instrument,
                            'tradingsymbol': row.tradingsymbol,
                            'signal_date': signal_date,
                            'entry_date': np.nan,
                            'entry_price': entry_price,
                            'num_shares': np.floor(backtest_config.get('capital_per_trade', 20000) * capital_multiplier / row.close) if row.close > 0 else 0,
                            'tp_price': tp_price,
                            'tp_thresh': tp_thresh,
                            'sl_price': sl_price,
                            'sl_thresh': sl_thresh,
                            'exit_date': pd.NaT,
                            'exit_price': np.nan,
                            'exit_reason': 'enter_today',
                            'pnl': np.nan,
                            'capital_multiplier': capital_multiplier,
                            'entry_signal_prob_1': row.entry_signal_prob_1,
                            'entry_signal_prob_2': row.entry_signal_prob_2,
                            'exit_signal_prob': row.exit_signal_prob,
                            'entry_source': 'tiered',
                        })

    # Wrap up open trades
    print(f"\nLogging {len(active_trades)} open trades at the end of the simulation (v3).")
    last_rows_map = precomputed_last_rows_map if precomputed_last_rows_map is not None else {instrument: group.iloc[-1] for instrument, group in df.groupby('instrument_token')}
    for instrument, trade in active_trades.items():
        last_row = last_rows_map.get(instrument)
        exit_reason = 'Active'
        if last_row is not None and backtest_config.get('exit_signal_integration', True) and last_row['exit_signal_prob'] > exit_trigger_threshold:
            exit_reason = 'exit_today'
        trade['exit_date'] = pd.NaT
        trade['exit_price'] = np.nan
        trade['exit_reason'] = exit_reason
        trade['pnl'] = np.nan
        trade_log.append(trade)

    print("Simulation (v3) complete.")
    return trade_log, debug_df


def calculate_performance_metrics(trade_log_df: pd.DataFrame, initial_capital: float = 500000) -> dict:
    print("\n--- Phase 3 (v3): Performance Calculation ---")
    unfiltered = trade_log_df.copy()
    pending_entry_count = (unfiltered['exit_reason'] == 'enter_today').sum()
    pending_exit_count = (unfiltered['exit_reason'] == 'exit_today').sum()
    active_on_close_count = (unfiltered['exit_reason'] == 'Active').sum()

    completed = unfiltered[~unfiltered['exit_reason'].isin(['enter_today', 'exit_today', 'Active'])].copy()
    print(f"Calculating metrics on {len(completed)} completed trades.")
    if completed.empty:
        return {
            'Total_Trades': 0, 'Win_Rate_%': 0, 'Total_PnL': 0, 'Profit_Factor': 0,
            'Avg_Trade_PnL': 0, 'Avg_Return_%': 0, 'Max_Drawdown_PnL': 0, 'Max_Drawdown_%': 0,
            'Best_Trade_PnL': 0, 'Worst_Trade_PnL': 0, 'Target_Hit_Count': 0, 'Stop_Loss_Count': 0,
            'Holding_Period_Count': 0, 'Exit_Signal_Count': 0,
            'Pending_Entry_Count': pending_entry_count, 'Pending_Exit_Count': pending_exit_count,
            'Active_On_Close_Count': active_on_close_count,
            'Avg_Holding_Period_Days': 0,
        }

    # Normalize dates and compute holding period days
    completed['entry_date'] = pd.to_datetime(completed['entry_date'])
    completed['exit_date'] = pd.to_datetime(completed['exit_date'])
    # EXIT_SIGNAL executes T+1 open; adjust effective exit date by +1 day
    exit_signal_mask = completed['exit_reason'] == 'EXIT_SIGNAL'
    completed['effective_exit_date'] = completed['exit_date'] + pd.to_timedelta(exit_signal_mask.astype(int), unit='D')
    # Use calendar days difference
    completed['holding_period_days'] = (
        completed['effective_exit_date'].dt.normalize() - completed['entry_date'].dt.normalize()
    ).dt.days

    completed['pnl_pct'] = (completed['pnl'] / completed['entry_price']) * 100
    total_trades = len(completed)
    wins = completed[completed['pnl'] > 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = completed['pnl'].sum()

    gross_profit = completed[completed['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(completed[completed['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    completed['exit_date'] = pd.to_datetime(completed['exit_date'])
    pnl_by_date = completed.groupby(completed['exit_date'].dt.date)['pnl'].sum()
    equity_curve = pnl_by_date.cumsum() + initial_capital
    peak = equity_curve.cummax()
    drawdown = peak - equity_curve
    max_drawdown_pnl = drawdown.max()
    max_drawdown_pct = (max_drawdown_pnl / peak[drawdown.idxmax()]) * 100 if len(peak) > 0 else 0

    return {
        'Total_Trades': total_trades,
        'Win_Rate_%': win_rate,
        'Total_PnL': total_pnl,
        'Profit_Factor': profit_factor,
        'Avg_Trade_PnL': completed['pnl'].mean(),
        'Avg_Return_%': completed['pnl_pct'].mean(),
        'Avg_Holding_Period_Days': completed['holding_period_days'].mean(),
        'Max_Drawdown_PnL': max_drawdown_pnl,
        'Max_Drawdown_%': max_drawdown_pct,
        'Best_Trade_PnL': completed['pnl'].max(),
        'Worst_Trade_PnL': completed['pnl'].min(),
        'Target_Hit_Count': (completed['exit_reason'] == 'TP').sum(),
        'Stop_Loss_Count': (completed['exit_reason'] == 'SL').sum(),
        'Holding_Period_Count': (completed['exit_reason'] == 'HP').sum(),
        'Exit_Signal_Count': (completed['exit_reason'] == 'EXIT_SIGNAL').sum(),
        'Pending_Entry_Count': pending_entry_count,
        'Pending_Exit_Count': pending_exit_count,
        'Active_On_Close_Count': active_on_close_count,
    }


def main():
    print("====== Starting Backtesting Engine (v3) ======")
    config = load_config()

    master_df = prepare_data_v3(config)
    trade_log, debug_df = run_simulation_v3(master_df, config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = 'reports'
    trades_dir = os.path.join(reports_dir, 'trades')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(trades_dir, exist_ok=True)

    debug_log_path = os.path.join(reports_dir, f'backtest_debug_log_v3_{timestamp}.csv')
    if debug_df is not None:
        debug_df.to_csv(debug_log_path, index=False)
        print(f"\nDebug log (v3) saved to {debug_log_path}")

    latest_timestamp = master_df['timestamp'].unique().max()
    latest_prob = master_df[master_df['timestamp'] == latest_timestamp]
    latest_prob.to_csv(os.path.join(trades_dir, 'daily_logs_v3.csv'), index=False)

    trade_log_df = pd.DataFrame(trade_log)
    if not trade_log_df.empty:
        trade_log_path = os.path.join(reports_dir, f'trade_log_v3_{timestamp}.csv')
        trade_log_df.to_csv(trade_log_path, index=False)
        trade_log_df.to_csv(os.path.join(trades_dir, 'daily_trades_v3.csv'), index=False)
        print(f"Trade log (v3) saved to {trade_log_path}")

    capital_per_trade = config.get('backtest_v3', {}).get('capital_per_trade', 50000)
    performance_metrics = calculate_performance_metrics(trade_log_df, initial_capital=capital_per_trade)

    print("\n--- Phase 4 (v3): Consolidated Reporting ---")
    try:
        summary_files = glob.glob(os.path.join(reports_dir, 'run_summary_*.csv'))
        latest_summary_file = max(summary_files, key=os.path.getctime)
        print(f"Loading latest run summary from: {latest_summary_file}")
        summary_df = pd.read_csv(latest_summary_file)
        backtest_results_df = pd.DataFrame([performance_metrics])
        final_report_df = pd.concat([summary_df.reset_index(drop=True), backtest_results_df.reset_index(drop=True)], axis=1)
    except (ValueError, FileNotFoundError):
        print("Could not find a recent run_summary.csv. Creating a new report with backtest and config details.")
        backtest_df = pd.DataFrame([performance_metrics])
        backtest_config = config.get('backtest_v3', {})
        for key, value in backtest_config.items():
            if not isinstance(value, (dict, list)):
                backtest_df[key] = value
        final_report_df = backtest_df

    final_report_path = os.path.join(reports_dir, f'backtest_report_v3_{timestamp}.csv')
    final_report_df.to_csv(final_report_path, index=False)
    print("\n--- FINAL BACKTEST REPORT (v3) ---")
    print(final_report_df.to_string())
    print(f"\nConsolidated backtest report (v3) saved to {final_report_path}")
    print("\n====== Backtesting Engine (v3) Finished ======")


if __name__ == '__main__':
    main()

