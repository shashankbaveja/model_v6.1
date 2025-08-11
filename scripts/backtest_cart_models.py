import os
import re
import sys
import glob
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

# Ensure local imports work when running as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(PROJECT_ROOT)
sys.path.append(SRC_DIR)
from src.data_pipeline import load_config as load_yaml_config
from src.trade_generator import run_simulation, calculate_performance_metrics

# For tradingsymbol enrichment (best-effort) — path is src/data_backfill/myKiteLib.py
try:
    from src.data_backfill.myKiteLib import system_initialization
except Exception:
    system_initialization = None


MODEL_FILENAME_REGEX = re.compile(
    r"^(?P<strategy>[A-Za-z0-9]+)_(?P<direction>up|down)_(?P<modeltype>[A-Za-z0-9]+)_L(?P<L>[0-9]+(?:\.[0-9]+)?)_TP(?P<TP>[0-9]+(?:\.[0-9]+)?)_SL(?P<SL>[0-9]+(?:\.[0-9]+)?)_model\.joblib$"
)


def parse_model_filename(filename: str) -> Dict[str, str]:
    base = os.path.basename(filename)
    match = MODEL_FILENAME_REGEX.match(base)
    if not match:
        raise ValueError(f"Model filename does not match expected pattern: {base}")
    parts = match.groupdict()
    parts["L"] = int(float(parts["L"]))
    parts["TP"] = float(parts["TP"])
    parts["SL"] = float(parts["SL"])
    return parts


def ensure_volatility(base_df: pd.DataFrame, feature_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    if "volatility_ewma_30d" in base_df.columns:
        return base_df
    if "volatility_ewma_30d" in feature_df.columns:
        vol_df = feature_df[["instrument_token", "timestamp", "volatility_ewma_30d"]].copy()
    else:
        def _compute_vol(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("timestamp").copy()
            daily_return = group["close"].pct_change()
            rolling_std = daily_return.rolling(window=30, min_periods=30).std()
            group["volatility_ewma_30d"] = rolling_std.ewm(span=30, adjust=False).mean()
            return group[["instrument_token", "timestamp", "volatility_ewma_30d"]]

        vol_df = (
            raw_df[["instrument_token", "timestamp", "close"]]
            .groupby("instrument_token", group_keys=False)
            .apply(_compute_vol)
        )
    return base_df.merge(vol_df, on=["instrument_token", "timestamp"], how="left")


def load_test_frames(processed_dir: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    strategy = "combined"
    feature_path = os.path.join(processed_dir, f"test_{strategy}_with_patterns_features.parquet")
    raw_path = os.path.join(processed_dir, "test_raw.parquet")

    feature_df = pd.read_parquet(feature_path)
    raw_df = pd.read_parquet(raw_path)

    # Normalize dtypes
    feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"]) if not np.issubdtype(feature_df["timestamp"].dtype, np.datetime64) else feature_df["timestamp"]
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"]) if not np.issubdtype(raw_df["timestamp"].dtype, np.datetime64) else raw_df["timestamp"]

    # Drop target columns if present
    feature_df = feature_df.drop(columns=["target_up", "target_down"], errors="ignore")

    # Build base frame with OHLCV
    base_keys = feature_df[["instrument_token", "timestamp"]]
    base_df = base_keys.merge(raw_df, on=["instrument_token", "timestamp"], how="left")
    base_df = ensure_volatility(base_df, feature_df, raw_df)

    # X for base models
    X_test = feature_df.drop(columns=["instrument_token", "timestamp"], errors="ignore")
    return feature_df, raw_df, base_df, X_test


def predict_all_base_models(model_paths: List[str], X_test: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    preds_df = pd.DataFrame(index=X_test.index)
    iterator = tqdm(model_paths, desc="Predicting base models") if verbose else model_paths
    for model_path in iterator:
        try:
            info = parse_model_filename(model_path)
            col_name = f"pred_{info['direction']}_L{info['L']}_TP{info['TP']}_SL{info['SL']}"
            model = joblib.load(model_path)
            preds_df[col_name] = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            print(f"Warning: Prediction failed for {os.path.basename(model_path)}: {e}")
            continue
    return preds_df


def enrich_tradingsymbol(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    if system_initialization is None:
        if verbose:
            print("DB access not available; skipping tradingsymbol enrichment.")
        return df
    try:
        systemDetails = system_initialization()
        token_name_list = systemDetails.run_query_full(
            "Select distinct instrument_token, tradingsymbol from kiteconnect.instruments_zerodha"
        )
        token_name = pd.DataFrame(token_name_list)
        if {"instrument_token", "tradingsymbol"}.issubset(token_name.columns):
            df = pd.merge(df, token_name[["instrument_token", "tradingsymbol"]], on="instrument_token", how="left")
        else:
            if verbose:
                print("Tradingsymbol columns not found in query result; skipping merge.")
    except Exception as e:
        if verbose:
            print(f"Tradingsymbol enrichment failed: {e}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Backtest each CART model as entry signal using static exit model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose progress output")
    args = parser.parse_args()

    config = load_yaml_config()
    processed_dir = "data/processed"
    signals_dir = "data/signals"
    models_dir = "models"
    cart_dir = os.path.join(models_dir, "cart_meta")
    os.makedirs(signals_dir, exist_ok=True)

    # Load test frames
    if args.verbose:
        print("[1/5] Loading test data...")
    f_test, r_test, base_test, X_test = load_test_frames(processed_dir)

    # Predict all base models
    if args.verbose:
        print("[2/5] Generating predictions for all base models (up + down)...")
    base_model_paths = sorted(glob.glob(os.path.join(models_dir, "*_model.joblib")))
    preds_test = predict_all_base_models(base_model_paths, X_test, verbose=args.verbose)
    pred_cols = sorted([c for c in preds_test.columns if c.startswith("pred_")])
    X_meta_test = preds_test[pred_cols].reset_index(drop=True)

    # Static exit probabilities
    if args.verbose:
        print("[3/5] Generating exit probabilities from static exit model...")
    exit_model_path = config.get("backtest", {}).get("exit_signal_model_path")
    if not exit_model_path or not os.path.exists(exit_model_path):
        raise FileNotFoundError("Static exit model path not found in config['backtest']['exit_signal_model_path'] or file missing")
    exit_model = joblib.load(exit_model_path)
    try:
        exit_signal_prob = exit_model.predict_proba(X_test)[:, 1]
    except Exception as e:
        raise RuntimeError(f"Exit model prediction failed: {e}")

    # Prepare base unified frame with keys + OHLCV + vol + exit prob + tradingsymbol
    unified_base = base_test.copy()
    unified_base = enrich_tradingsymbol(unified_base, verbose=args.verbose)
    unified_base["exit_signal_prob"] = exit_signal_prob

    # Iterate over CART entry models
    if args.verbose:
        print("[4/5] Backtesting each CART model as entry signal...")
    cart_paths = sorted(glob.glob(os.path.join(cart_dir, "*.joblib")))
    if not cart_paths:
        print(f"No CART models found under {cart_dir}")
        sys.exit(0)

    results = []
    for cart_path in tqdm(cart_paths, desc="Backtesting CARTs") if args.verbose else cart_paths:
        cart_name = os.path.splitext(os.path.basename(cart_path))[0]

        # Predict binary entry signals from CART
        try:
            cart = joblib.load(cart_path)
            # Align feature count if necessary (pad/truncate names handled at train time)
            entry_labels = cart.predict(X_meta_test)
            entry_signal_prob = entry_labels.astype(float)
        except Exception as e:
            print(f"Skipping {cart_name}: CART prediction failed ({e})")
            continue

        unified_df = unified_base.copy()
        unified_df["entry_signal_prob"] = entry_signal_prob

        # Save unified signals for traceability
        out_signals_path = os.path.join(signals_dir, f"unified_signals_{cart_name}.csv")
        try:
            unified_df.to_csv(out_signals_path, index=False)
        except Exception as e:
            print(f"Warning: Could not save unified signals for {cart_name}: {e}")

        # Run simulation and collect metrics
        try:
            trade_log, debug_df = run_simulation(unified_df, config)
            trade_log_df = pd.DataFrame(trade_log)
            capital_per_trade = config.get('backtest',{}).get('capital_per_trade', 50000)
            metrics = calculate_performance_metrics(trade_log_df, initial_capital=capital_per_trade)
        except Exception as e:
            print(f"Backtest failed for {cart_name}: {e}")
            continue

        metrics_row = {"cart_model": cart_name}
        metrics_row.update(metrics)
        results.append(metrics_row)

    # Save aggregated results
    if results:
        results_df = pd.DataFrame(results)
        results_path = os.path.join(signals_dir, "cart_backtest_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"[5/5] Backtesting complete. Results saved to: {results_path}")
    else:
        print("No results to save (no successful backtests).")


if __name__ == "__main__":
    main()

