import os
import re
import sys
import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

# Ensure local imports work when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import load_config


MODEL_FILENAME_REGEX = re.compile(
    r"^(?P<strategy>[A-Za-z0-9]+)_(?P<direction>up|down)_(?P<modeltype>[A-Za-z0-9]+)_L(?P<L>[0-9]+(?:\.[0-9]+)?)_TP(?P<TP>[0-9]+(?:\.[0-9]+)?)_SL(?P<SL>[0-9]+(?:\.[0-9]+)?)_model\.joblib$"
)


def parse_model_filename(filename: str) -> Dict[str, str]:
    """Parse a model filename into its components.

    Expected format:
      {strategy}_{direction}_{modeltype}_L{lookahead}_TP{tp}_SL{sl}_model.joblib
    """
    base = os.path.basename(filename)
    match = MODEL_FILENAME_REGEX.match(base)
    if not match:
        raise ValueError(f"Model filename does not match expected pattern: {base}")
    parts = match.groupdict()
    # Convert numeric fields to proper types
    parts["L"] = int(float(parts["L"]))
    parts["TP"] = float(parts["TP"])
    parts["SL"] = float(parts["SL"])
    return parts


def compute_directional_volatility_target(
    df: pd.DataFrame,
    lookahead_candles: int,
    tp_multiplier: float,
    sl_multiplier: float,
    direction: str,
) -> pd.Series:
    """Compute a binary target for one direction based on volatility bands.

    For each time t, define profit_target = close_t * (1 + vol_ewma_t * tp_multiplier)
                                 stop_loss     = close_t * (1 - vol_ewma_t * tp_multiplier)
    Look forward up to lookahead_candles on the next candles (t+1 .. t+lookahead)
    and mark 1 if the first event is in the requested direction, else 0.

    The input df must contain: instrument_token, timestamp, close, volatility_ewma_30d
    """
    required_cols = {"instrument_token", "timestamp", "close", "volatility_ewma_30d"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for target computation: {sorted(missing)}")

    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")

    # Work on a copy to avoid mutating caller
    work_df = df[["instrument_token", "timestamp", "close", "volatility_ewma_30d"]].copy()

    # Ensure proper dtypes
    if not np.issubdtype(work_df["timestamp"].dtype, np.datetime64):
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"])  # best-effort

    # Sort within instrument groups to maintain chronological order
    work_df.sort_values(["instrument_token", "timestamp"], inplace=True)

    targets: List[int] = []

    # We compute per instrument to avoid lookahead across instruments
    grouped = work_df.groupby("instrument_token", sort=False)
    for _, group in grouped:
        closes = group["close"].to_numpy()
        vol = group["volatility_ewma_30d"].to_numpy()
        n = len(group)

        profit_levels = closes * (1.0 + vol * tp_multiplier)
        # Intentional: stop loss uses the TP multiplier (to match training target generation)
        stop_levels = closes * (1.0 - vol * tp_multiplier)

        group_targets = np.zeros(n, dtype=int)

        if lookahead_candles > 0 and n > 1:
            # Match training loop bounds exactly: range(n - lookahead)
            for i in range(max(0, n - lookahead_candles)):
                end_idx = min(n, i + 1 + lookahead_candles)
                if i + 1 >= end_idx:
                    continue
                window = closes[i + 1 : end_idx]

                # Find first hit index for TP and SL, relative to window start
                tp_hit_rel_idx = np.argmax(window >= profit_levels[i]) if np.any(window >= profit_levels[i]) else -1
                sl_hit_rel_idx = np.argmax(window <= stop_levels[i]) if np.any(window <= stop_levels[i]) else -1

                if tp_hit_rel_idx == -1 and sl_hit_rel_idx == -1:
                    continue

                if tp_hit_rel_idx == -1:
                    first_event = "sl"
                elif sl_hit_rel_idx == -1:
                    first_event = "tp"
                else:
                    first_event = "tp" if tp_hit_rel_idx < sl_hit_rel_idx else "sl"

                if (direction == "up" and first_event == "tp") or (direction == "down" and first_event == "sl"):
                    group_targets[i] = 1

        targets.extend(group_targets.tolist())

    return pd.Series(targets, index=work_df.index)


def build_consolidated_predictions(
    models_dir: str = "models",
    processed_dir: str = "data/processed",
    output_csv_path: str = "data/signals/all_model_signals.csv",
) -> str:
    """Iterate all models, generate predictions and per-model targets, and save a consolidated CSV.

    Returns the path to the saved CSV.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    model_paths = sorted(glob.glob(os.path.join(models_dir, "*_model.joblib")))
    if not model_paths:
        raise FileNotFoundError(f"No model files found in: {models_dir}")

    # Derive strategy from the first model (user noted single strategy)
    first_model_info = parse_model_filename(model_paths[0])
    strategy = first_model_info["strategy"]

    # Load datasets once
    feature_data_path = os.path.join(processed_dir, f"test_{strategy}_with_patterns_features.parquet")
    raw_test_path = os.path.join(processed_dir, "test_raw.parquet")

    feature_df = pd.read_parquet(feature_data_path)
    raw_test_df = pd.read_parquet(raw_test_path)

    # Ensure keys are present
    required_keys = {"instrument_token", "timestamp"}
    if not required_keys.issubset(feature_df.columns) or not required_keys.issubset(raw_test_df.columns):
        raise ValueError("Both feature and raw data must contain 'instrument_token' and 'timestamp' columns.")

    if not np.issubdtype(feature_df["timestamp"].dtype, np.datetime64):
        feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"])  # best-effort
    if not np.issubdtype(raw_test_df["timestamp"].dtype, np.datetime64):
        raw_test_df["timestamp"] = pd.to_datetime(raw_test_df["timestamp"])  # best-effort

    # Drop any existing targets in features
    feature_df = feature_df.drop(columns=["target_up", "target_down"], errors="ignore")

    # Build base frame in the order of feature_df rows (to align with X_test and predictions)
    base_keys = feature_df[["instrument_token", "timestamp"]].copy()
    consolidated_df = base_keys.merge(
        raw_test_df,
        on=["instrument_token", "timestamp"],
        how="left",
        suffixes=("", ""),
    )
    # Ensure volatility feature is present for target computation
    if "volatility_ewma_30d" not in consolidated_df.columns:
        if "volatility_ewma_30d" in feature_df.columns:
            vol_df = feature_df[["instrument_token", "timestamp", "volatility_ewma_30d"]].copy()
        else:
            # Fallback: compute volatility from raw close prices per instrument
            def _compute_vol(group: pd.DataFrame) -> pd.DataFrame:
                group = group.sort_values("timestamp").copy()
                daily_return = group["close"].pct_change()
                rolling_std = daily_return.rolling(window=30, min_periods=30).std()
                group["volatility_ewma_30d"] = rolling_std.ewm(span=30, adjust=False).mean()
                return group[["instrument_token", "timestamp", "volatility_ewma_30d"]]

            vol_df = (
                raw_test_df[["instrument_token", "timestamp", "close"]]
                .groupby("instrument_token", group_keys=False)
                .apply(_compute_vol)
            )

        consolidated_df = consolidated_df.merge(
            vol_df,
            on=["instrument_token", "timestamp"],
            how="left",
        )

    # Prepare X matrix for prediction
    X_test = feature_df.drop(columns=["instrument_token", "timestamp"], errors="ignore")

    # Iterate models and add columns
    for model_path in model_paths:
        try:
            info = parse_model_filename(model_path)
        except ValueError as e:
            # Skip unparseable files
            print(f"Skipping file (name pattern mismatch): {model_path} -- {e}")
            continue

        direction = info["direction"]
        lookahead = info["L"]
        tp = info["TP"]
        sl = info["SL"]

        target_col_name = f"target_{direction}_L{lookahead}_TP{tp}_SL{sl}"
        pred_col_name = f"pred_{direction}_L{lookahead}_TP{tp}_SL{sl}"

        # Compute per-model direction-specific target
        try:
            target_series = compute_directional_volatility_target(
                df=consolidated_df,
                lookahead_candles=lookahead,
                tp_multiplier=tp,
                sl_multiplier=sl,
                direction=direction,
            )
            target_series = target_series.reset_index(drop=True)
            consolidated_df[target_col_name] = target_series.astype(int)
        except Exception as e:
            print(f"Warning: Failed to compute target for {os.path.basename(model_path)}: {e}")
            consolidated_df[target_col_name] = np.nan

        # Load model and predict
        try:
            model = joblib.load(model_path)
            proba = model.predict_proba(X_test)[:, 1]
            consolidated_df[pred_col_name] = proba
        except Exception as e:
            print(f"Warning: Failed to predict for {os.path.basename(model_path)}: {e}")
            consolidated_df[pred_col_name] = np.nan

    # Save consolidated output
    consolidated_df.to_csv(output_csv_path, index=False)
    return output_csv_path


def process_consolidated_signals(
    input_csv_path: str = "data/signals/all_model_signals.csv",
    output_csv_path: str = "data/signals/all_model_signals_percentiles.csv",
) -> str:
    """Process the consolidated signals CSV and compute per-model percentile bucket stats.

    For each model's prediction column (pred_*), this computes decile bins based on the
    prediction distribution and reports:
      - coverage_pct: percentage of total positives captured within the bin
      - precision_pct: percentage of bin rows that are positives (positive rate)
      - counts and numeric bounds per bin
    The matching target column is inferred by replacing the 'pred_' prefix with 'target_'.
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)

    # Identify model prediction columns
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        raise ValueError("No prediction columns (pred_*) found in the consolidated CSV.")

    results: List[Dict[str, object]] = []

    # Ensure numeric dtype for predictions and targets
    for pred_col in pred_cols:
        suffix = pred_col[len("pred_"):]
        target_col = f"target_{suffix}"
        if target_col not in df.columns:
            print(f"Warning: Matching target column not found for {pred_col}. Expected: {target_col}. Skipping.")
            continue

        # Prepare series
        pred = pd.to_numeric(df[pred_col], errors="coerce")
        target_raw = df[target_col]
        # Coerce target to 0/1, treat non-finite as NaN then drop
        target = pd.to_numeric(target_raw, errors="coerce").fillna(0).astype(int)

        valid_mask = pred.notna()
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]

        total_targets = int(target_valid.sum())
        total_rows = int(valid_mask.sum())

        # Compute decile cutoffs and percentile bins robustly
        # Rank-based bins for assignment to avoid quantile duplication errors
        ranks = pred_valid.rank(pct=True, method="average")  # in (0,1]
        # Map to deciles 1..10
        deciles = np.clip(np.ceil(ranks * 10).astype(int), 1, 10)

        # Numeric bounds for each decile from quantiles of prediction
        quantiles = np.quantile(pred_valid, np.linspace(0, 1, 11))

        for d in range(1, 11):
            bin_mask = deciles == d
            bin_total = int(bin_mask.sum())
            bin_targets = int(target_valid[bin_mask].sum()) if bin_total > 0 else 0

            coverage_pct = (100.0 * bin_targets / total_targets) if total_targets > 0 else 0.0
            precision_pct = (100.0 * bin_targets / bin_total) if bin_total > 0 else 0.0

            lower_bound = float(quantiles[d - 1])
            upper_bound = float(quantiles[d])

            results.append(
                {
                    "model": pred_col,
                    "target": target_col,
                    "percentile_range": f"{(d-1)*10}-{d*10}",
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "bin_total": bin_total,
                    "bin_targets": bin_targets,
                    "coverage_pct": coverage_pct,
                    "precision_pct": precision_pct,
                    "total_rows": total_rows,
                    "total_targets": total_targets,
                }
            )

    if not results:
        raise ValueError("No valid model prediction/target pairs found to process.")

    report_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    report_df.to_csv(output_csv_path, index=False)
    return output_csv_path


def main():
    """Entry point: generate consolidated predictions for all models."""
    # Load config primarily to ensure environment consistency; parameters come from model names
    _ = load_config()

    output_path = build_consolidated_predictions(
        models_dir="models",
        processed_dir="data/processed",
        output_csv_path="data/signals/all_model_signals.csv",
    )
    print(f"Consolidated signals saved to: {output_path}")

    # By default, process the consolidated file into percentile analysis
    processed_path = process_consolidated_signals(
        input_csv_path=output_path,
        output_csv_path="data/signals/all_model_signals_percentiles.csv",
    )
    print(f"Percentile analysis saved to: {processed_path}")


if __name__ == "__main__":
    main()

