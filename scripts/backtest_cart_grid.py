import os
import re
import sys
import glob
import argparse
from typing import Dict, List, Tuple

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

# For tradingsymbol enrichment
try:
    from src.data_backfill.myKiteLib import system_initialization
except Exception:
    system_initialization = None


BASE_MODEL_REGEX = re.compile(
    r"^(?P<strategy>[A-Za-z0-9]+)_(?P<direction>up|down)_(?P<modeltype>[A-Za-z0-9]+)_L(?P<L>[0-9]+(?:\.[0-9]+)?)_TP(?P<TP>[0-9]+(?:\.[0-9]+)?)_SL(?P<SL>[0-9]+(?:\.[0-9]+)?)_model\.joblib$"
)

CART_MODEL_REGEX = re.compile(
    r"^cart_meta_up_L(?P<L>[0-9]+)_TP(?P<TP>[0-9]+(?:\.[0-9]+)?)_SL(?P<SL>[0-9]+(?:\.[0-9]+)?)\.joblib$"
)


def parse_base_model_filename(path: str) -> Dict[str, str]:
    name = os.path.basename(path)
    m = BASE_MODEL_REGEX.match(name)
    if not m:
        raise ValueError(f"Base model filename does not match pattern: {name}")
    parts = m.groupdict()
    parts["L"] = int(float(parts["L"]))
    parts["TP"] = float(parts["TP"])
    parts["SL"] = float(parts["SL"])
    return parts


def parse_cart_filename(path: str) -> Tuple[int, float, float, str]:
    name = os.path.basename(path)
    m = CART_MODEL_REGEX.match(name)
    if not m:
        raise ValueError(f"CART model filename does not match pattern: {name}")
    L = int(m.group("L"))
    TP = float(m.group("TP"))
    SL_in_name = float(m.group("SL"))
    return L, TP, SL_in_name, name[:-7]  # base name without .joblib


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


def load_test_frames(processed_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy = "combined"
    feature_path = os.path.join(processed_dir, f"test_{strategy}_with_patterns_features.parquet")
    raw_path = os.path.join(processed_dir, "test_raw.parquet")

    feature_df = pd.read_parquet(feature_path)
    raw_df = pd.read_parquet(raw_path)

    feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"]) if not np.issubdtype(feature_df["timestamp"].dtype, np.datetime64) else feature_df["timestamp"]
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"]) if not np.issubdtype(raw_df["timestamp"].dtype, np.datetime64) else raw_df["timestamp"]

    feature_df = feature_df.drop(columns=["target_up", "target_down"], errors="ignore")

    base_keys = feature_df[["instrument_token", "timestamp"]]
    base_df = base_keys.merge(raw_df, on=["instrument_token", "timestamp"], how="left")
    base_df = ensure_volatility(base_df, feature_df, raw_df)

    X_test = feature_df.drop(columns=["instrument_token", "timestamp"], errors="ignore")
    return feature_df, raw_df, base_df, X_test


def predict_all_base_models(model_paths: List[str], X_test: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    preds_df = pd.DataFrame(index=X_test.index)
    iterator = tqdm(model_paths, desc="Predicting base models") if verbose else model_paths
    for model_path in iterator:
        try:
            info = parse_base_model_filename(model_path)
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


def update_parameters_yml(lookahead: int, tp_multiplier: float, sl_multiplier: float, config_path: str = 'config/parameters.yml') -> bool:
    """Update parameters.yml target_generation values in-place using a simple line-edit approach.

    This mirrors the behavior of scripts/train_all_models.py to avoid YAML reformatting.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Known indexes based on file structure
        lines[16] = f"  lookahead_candles: {lookahead}\n"
        lines[17] = f"  volatility_tp_multipler: {tp_multiplier}\n"
        lines[18] = f"  volatility_sl_multipler: {sl_multiplier}\n"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        return True
    except Exception as e:
        print(f"Warning: Failed to update parameters.yml: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Grid backtest CART entry models vs DOWN exit models over SL values")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose progress output")
    args = parser.parse_args()

    config = load_yaml_config()
    processed_dir = "data/processed"
    signals_dir = "data/signals"
    models_dir = "models"
    cart_dir = os.path.join(models_dir, "cart_meta")
    os.makedirs(signals_dir, exist_ok=True)

    # Load test frames once
    if args.verbose:
        print("[1/6] Loading test data...")
    f_test, r_test, base_test, X_test = load_test_frames(processed_dir)

    # Predict all base models for CART inputs
    if args.verbose:
        print("[2/6] Generating predictions for all base models (up + down) on test...")
    base_model_paths = sorted(glob.glob(os.path.join(models_dir, "*_model.joblib")))
    # Keep only 'combined' strategy files for safety
    base_model_paths = [p for p in base_model_paths if os.path.basename(p).startswith("combined_")]
    preds_test = predict_all_base_models(base_model_paths, X_test, verbose=args.verbose)
    pred_cols = sorted([c for c in preds_test.columns if c.startswith("pred_")])
    if not pred_cols:
        print("No base model predictions were generated; aborting.")
        sys.exit(1)
    X_meta_test = preds_test[pred_cols].reset_index(drop=True)

    # Discover exit DOWN models
    exit_model_paths = [
        p for p in base_model_paths
        if BASE_MODEL_REGEX.match(os.path.basename(p)) and parse_base_model_filename(p)["direction"] == "down"
    ]
    if args.verbose:
        print(f"[3/6] Found {len(exit_model_paths)} DOWN exit models.")

    # Cache exit predictions per exit model
    exit_preds_cache: Dict[str, np.ndarray] = {}

    # Discover CART entry models
    cart_paths = sorted(glob.glob(os.path.join(cart_dir, "*.joblib")))
    if not cart_paths:
        print(f"No CART models found under {cart_dir}")
        sys.exit(0)

    results: List[Dict[str, object]] = []

    if args.verbose:
        print("[4/6] Running grid backtests (CART x EXIT x SL)...")

    for cart_path in tqdm(cart_paths, desc="CART models") if args.verbose else cart_paths:
        try:
            L_from_cart, TP_from_cart, _SL_in_name, cart_base = parse_cart_filename(cart_path)
        except ValueError as e:
            print(f"Skipping CART {os.path.basename(cart_path)}: {e}")
            continue

        # Load CART model
        try:
            cart_model = joblib.load(cart_path)
        except Exception as e:
            print(f"Skipping CART {cart_base}: failed to load ({e})")
            continue

        # Predict entry labels once per CART
        try:
            entry_labels = cart_model.predict(X_meta_test)
            entry_signal_prob = entry_labels.astype(float)
        except Exception as e:
            print(f"Skipping CART {cart_base}: prediction failed ({e})")
            continue

        for exit_path in tqdm(exit_model_paths, desc=f"Exit models for {cart_base}") if args.verbose else exit_model_paths:
            exit_name = os.path.basename(exit_path)

            # Get or compute exit predictions
            if exit_name in exit_preds_cache:
                exit_signal_prob = exit_preds_cache[exit_name]
            else:
                try:
                    exit_model = joblib.load(exit_path)
                    exit_signal_prob = exit_model.predict_proba(X_test)[:, 1]
                    exit_preds_cache[exit_name] = exit_signal_prob
                except Exception as e:
                    print(f"Skipping exit {exit_name}: prediction failed ({e})")
                    continue

            for SL in [2.0, 3.0, 4.0, 5.0]:
                # Update parameters.yml and in-memory config to reflect (L, TP, SL)
                update_parameters_yml(L_from_cart, TP_from_cart, SL)
                config['target_generation']['lookahead_candles'] = L_from_cart
                config['target_generation']['volatility_tp_multipler'] = TP_from_cart
                config['target_generation']['volatility_sl_multipler'] = SL

                unified_df = base_test.copy()
                unified_df = enrich_tradingsymbol(unified_df, verbose=args.verbose)
                unified_df['entry_signal_prob'] = entry_signal_prob
                unified_df['exit_signal_prob'] = exit_signal_prob

                # Save unified signals for traceability
                signals_name = f"unified_signals_{cart_base}__{os.path.splitext(exit_name)[0]}__SL{SL}.csv"
                out_signals_path = os.path.join(signals_dir, signals_name)
                try:
                    unified_df.to_csv(out_signals_path, index=False)
                except Exception as e:
                    print(f"Warning: Could not save unified signals ({signals_name}): {e}")

                # Backtest
                try:
                    trade_log, debug_df = run_simulation(unified_df, config)
                    trade_log_df = pd.DataFrame(trade_log)
                    capital_per_trade = config.get('backtest',{}).get('capital_per_trade', 50000)
                    metrics = calculate_performance_metrics(trade_log_df, initial_capital=capital_per_trade)
                except Exception as e:
                    print(f"Backtest failed for {cart_base} with exit {exit_name} SL {SL}: {e}")
                    continue

                row = {
                    'cart_model': cart_base,
                    'exit_model': os.path.splitext(exit_name)[0],
                    'lookahead_L': L_from_cart,
                    'tp_multiplier': TP_from_cart,
                    'sl_multiplier': SL,
                }
                row.update(metrics)
                results.append(row)

    # Save results
    results_path = os.path.join(signals_dir, 'cart_pair_backtest_results.csv')
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Grid backtesting complete. Results saved to: {results_path}")


if __name__ == '__main__':
    main()

