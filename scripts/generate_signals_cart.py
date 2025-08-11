import os
import re
import sys
import glob
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

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

    work_df = df[["instrument_token", "timestamp", "close", "volatility_ewma_30d"]].copy()

    if not np.issubdtype(work_df["timestamp"].dtype, np.datetime64):
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"])  # best-effort

    work_df.sort_values(["instrument_token", "timestamp"], inplace=True)

    targets: List[int] = []

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


def ensure_volatility(
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure base_df has volatility_ewma_30d; merge from features or compute from raw."""
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

    merged = base_df.merge(vol_df, on=["instrument_token", "timestamp"], how="left")
    return merged


def load_partition_frames(
    processed_dir: str,
    strategy: str,
    partition: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features, raw, base frame and X for a partition (train/validation)."""
    feature_path = os.path.join(processed_dir, f"{partition}_{strategy}_with_patterns_features.parquet")
    raw_path = os.path.join(processed_dir, f"{partition}_raw.parquet")

    feature_df = pd.read_parquet(feature_path)
    raw_df = pd.read_parquet(raw_path)

    # Standardize timestamps
    feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"]) if not np.issubdtype(feature_df["timestamp"].dtype, np.datetime64) else feature_df["timestamp"]
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"]) if not np.issubdtype(raw_df["timestamp"].dtype, np.datetime64) else raw_df["timestamp"]

    # Drop any existing targets from features
    feature_df = feature_df.drop(columns=["target_up", "target_down"], errors="ignore")

    base_keys = feature_df[["instrument_token", "timestamp"]].copy()
    base_df = base_keys.merge(
        raw_df,
        on=["instrument_token", "timestamp"],
        how="left",
    )
    base_df = ensure_volatility(base_df, feature_df, raw_df)

    X = feature_df.drop(columns=["instrument_token", "timestamp"], errors="ignore")
    return feature_df, raw_df, base_df, X


def predict_all_models(model_paths: List[str], X: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    preds_df = pd.DataFrame(index=X.index)
    iterator = tqdm(model_paths, desc="Predicting base models") if verbose else model_paths
    for model_path in iterator:
        try:
            info = parse_model_filename(model_path)
        except ValueError as e:
            print(f"Skipping model (name pattern mismatch): {model_path} -- {e}")
            continue

        col_name = f"pred_{info['direction']}_L{info['L']}_TP{info['TP']}_SL{info['SL']}"
        try:
            model = joblib.load(model_path)
            preds_df[col_name] = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"Warning: Prediction failed for {os.path.basename(model_path)}: {e}")
            preds_df[col_name] = np.nan
    return preds_df


def compute_up_targets_for_partition(
    base_df: pd.DataFrame,
    up_model_infos: List[Dict[str, str]],
) -> pd.DataFrame:
    targets_df = pd.DataFrame(index=base_df.index)
    for info in tqdm(up_model_infos, desc="Computing UP targets"):
        col_name = f"target_up_L{info['L']}_TP{info['TP']}_SL{info['SL']}"
        try:
            s = compute_directional_volatility_target(
                df=base_df,
                lookahead_candles=info["L"],
                tp_multiplier=info["TP"],
                sl_multiplier=info["SL"],  # not used intentionally in the logic
                direction="up",
            ).reset_index(drop=True)
            targets_df[col_name] = s.astype(int)
        except Exception as e:
            print(f"Warning: Target computation failed for {col_name}: {e}")
            targets_df[col_name] = np.nan
    return targets_df


def deep_tune_and_train_cart(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    verbose: bool = False,
    tune_subsample_frac: float = 0.2,
) -> Tuple[DecisionTreeClassifier, Dict[str, float]]:
    """Deep tuning via CV over max_depth, min_samples_leaf, and ccp_alpha from cost-complexity path."""
    depth_grid = [2, 3, 4, 5]
    leaf_grid = [5000, 10000]

    # Generate candidate alphas
    base_tree = DecisionTreeClassifier(random_state=random_state, class_weight='balanced')
    try:
        path = base_tree.cost_complexity_pruning_path(X, y)
        alphas_full = np.unique(path.ccp_alphas)
        # Subsample alphas to a manageable size if very long
        # Aggressive subsample of alphas for speed
        if len(alphas_full) > 10:
            idx = np.linspace(0, len(alphas_full) - 1, 10).round().astype(int)
            alphas = list(alphas_full[idx])
        else:
            alphas = list(alphas_full)
    except Exception:
        alphas = [0.0]

    best_auc = -np.inf
    best_params: Dict[str, float] = {"max_depth": 4, "min_samples_leaf": 10000, "ccp_alpha": 0.0}
    best_model = None

    # Optional subsample for tuning speed
    if 0 < tune_subsample_frac < 1.0:
        # Stratified subsample indexes
        rng = np.random.RandomState(random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        pos_keep = rng.choice(pos_idx, size=max(1, int(len(pos_idx) * tune_subsample_frac)), replace=False) if len(pos_idx) > 0 else np.array([], dtype=int)
        neg_keep = rng.choice(neg_idx, size=max(1, int(len(neg_idx) * tune_subsample_frac)), replace=False) if len(neg_idx) > 0 else np.array([], dtype=int)
        keep_idx = np.concatenate([pos_keep, neg_keep])
        X_tune = X.iloc[keep_idx]
        y_tune = y.iloc[keep_idx]
        if verbose:
            print(f"  Tuning on subsample: {len(keep_idx)} rows ({tune_subsample_frac*100:.0f}% of data)")
    else:
        X_tune, y_tune = X, y

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    total_combos = len(depth_grid) * len(leaf_grid) * len(alphas)
    prog = tqdm(total=total_combos, desc="CART tuning", leave=False) if verbose else None
    for d in depth_grid:
        for leaf in leaf_grid:
            for alpha in alphas:
                model = DecisionTreeClassifier(
                    random_state=random_state,
                    class_weight='balanced',
                    max_depth=d,
                    min_samples_leaf=leaf,
                    ccp_alpha=alpha,
                )
                aucs = []
                try:
                    for train_idx, valid_idx in cv.split(X_tune, y_tune):
                        X_tr, X_va = X_tune.iloc[train_idx], X_tune.iloc[valid_idx]
                        y_tr, y_va = y_tune.iloc[train_idx], y_tune.iloc[valid_idx]
                        model.fit(X_tr, y_tr)
                        proba = model.predict_proba(X_va)[:, 1]
                        if len(np.unique(y_va)) < 2:
                            continue
                        aucs.append(roc_auc_score(y_va, proba))
                except Exception:
                    continue

                if aucs:
                    mean_auc = float(np.mean(aucs))
                    if mean_auc > best_auc:
                        best_auc = mean_auc
                        best_params = {"max_depth": d, "min_samples_leaf": leaf, "ccp_alpha": alpha}
                        if verbose:
                            print(f"  New best AUC={mean_auc:.4f} with params: depth={d}, leaf={leaf}, ccp_alpha={alpha:.6g}")
                        best_model = DecisionTreeClassifier(
                            random_state=random_state,
                            class_weight='balanced',
                            **best_params,
                        )
                if prog:
                    prog.update(1)

    if prog:
        prog.close()

    if best_model is None:
        best_model = DecisionTreeClassifier(random_state=random_state, class_weight='balanced', **best_params)
    best_model.fit(X, y)
    return best_model, best_params


def evaluate_model(model: DecisionTreeClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    out = {"auc": np.nan, "accuracy": np.nan, "precision": np.nan, "recall": np.nan}
    try:
        proba = model.predict_proba(X)[:, 1]
        if len(np.unique(y)) >= 2:
            out["auc"] = float(roc_auc_score(y, proba))
        pred = (proba >= 0.5).astype(int)
        out["accuracy"] = float(accuracy_score(y, pred))
        out["precision"] = float(precision_score(y, pred, zero_division=0))
        out["recall"] = float(recall_score(y, pred, zero_division=0))
    except Exception:
        pass
    return out


def main():
    parser = argparse.ArgumentParser(description="Train CART meta-models on base model predictions")
    # Deep tuning is now the default; the following flags are retained for compatibility but ignored.
    parser.add_argument("--fast", action="store_true", help="(Ignored) Fast mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose progress output")
    parser.add_argument("--max-depth", type=int, default=4, help="(Ignored) Fixed max_depth for CART")
    parser.add_argument("--min-samples-leaf", type=int, default=10000, help="(Ignored) Fixed min_samples_leaf for CART")
    parser.add_argument("--ccp-alpha", type=float, default=0.0, help="(Ignored) Fixed ccp_alpha for CART")
    args = parser.parse_args()

    config = load_config()

    models_dir = "models"
    processed_dir = "data/processed"
    signals_dir = "data/signals"
    os.makedirs(signals_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, "cart_meta"), exist_ok=True)

    if args.verbose:
        print("[1/6] Discovering models...")
    model_paths = sorted(glob.glob(os.path.join(models_dir, "*_model.joblib")))
    if not model_paths:
        raise FileNotFoundError(f"No model files found in: {models_dir}")

    # Infer strategy from first model (single-strategy assumption)
    first_info = parse_model_filename(model_paths[0])
    strategy = first_info["strategy"]
    if args.verbose:
        print(f"  Strategy inferred: {strategy}")

    if args.verbose:
        print("[2/6] Loading train/validation partitions...")
    f_train, r_train, base_train, X_train = load_partition_frames(processed_dir, strategy, "train")
    f_val, r_val, base_val, X_val = load_partition_frames(processed_dir, strategy, "validation")

    if args.verbose:
        print("[3/6] Generating base-model predictions (train)...")
    preds_train = predict_all_models(model_paths, X_train, verbose=args.verbose)
    if args.verbose:
        print("[3/6] Generating base-model predictions (validation)...")
    preds_val = predict_all_models(model_paths, X_val, verbose=args.verbose)

    if args.verbose:
        print("[4/6] Preparing UP targets for each UP config...")
    # Restrict to the specified 4 UP configurations
    allowed_up = {
        (7, 4.0, 2.0),
        (15, 5.0, 2.0),
        (15, 4.0, 2.0),
        (10, 4.0, 2.0),
    }
    up_model_infos = []
    for p in model_paths:
        try:
            info = parse_model_filename(p)
            if info["direction"] == "up" and (info["L"], info["TP"], info["SL"]) in allowed_up:
                up_model_infos.append(info)
        except ValueError:
            continue

    y_train_df = compute_up_targets_for_partition(base_train, up_model_infos)
    y_val_df = compute_up_targets_for_partition(base_val, up_model_infos)

    if args.verbose:
        print("[5/6] Training CART meta-models...")

    pred_cols_train = [c for c in preds_train.columns if c.startswith("pred_")]
    pred_cols_val = [c for c in preds_val.columns if c.startswith("pred_")]
    common_pred_cols = sorted(list(set(pred_cols_train).intersection(set(pred_cols_val))))
    X_meta_train = preds_train[common_pred_cols].reset_index(drop=True)
    X_meta_val = preds_val[common_pred_cols].reset_index(drop=True)

    results: List[Dict[str, object]] = []
    target_iter = tqdm(list(y_train_df.columns), desc="Training CARTs") if args.verbose else y_train_df.columns
    for target_col in target_iter:
        y_tr = y_train_df[target_col].fillna(0).astype(int)
        y_va = y_val_df[target_col].fillna(0).astype(int)

        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            if args.verbose:
                print(f"  Skipping {target_col}: degenerate labels in training.")
            continue

        # Deep tuning
        model, params = deep_tune_and_train_cart(
            X_meta_train,
            y_tr,
            verbose=args.verbose,
        )

        train_metrics = evaluate_model(model, X_meta_train, y_tr)
        val_metrics = evaluate_model(model, X_meta_val, y_va)

        model_filename = target_col.replace("target_", "cart_meta_") + ".joblib"
        model_path = os.path.join(models_dir, "cart_meta", model_filename)
        try:
            joblib.dump(model, model_path)
        except Exception as e:
            print(f"Warning: Could not save CART model {model_filename}: {e}")

        results.append(
            {
                "target": target_col,
                "num_train_rows": int(len(X_meta_train)),
                "num_val_rows": int(len(X_meta_val)),
                "pos_train": int(y_tr.sum()),
                "pos_val": int(y_va.sum()),
                "params_max_depth": params.get("max_depth"),
                "params_min_samples_leaf": params.get("min_samples_leaf"),
                "params_ccp_alpha": params.get("ccp_alpha"),
                "train_auc": train_metrics.get("auc"),
                "train_accuracy": train_metrics.get("accuracy"),
                "train_precision": train_metrics.get("precision"),
                "train_recall": train_metrics.get("recall"),
                "val_auc": val_metrics.get("auc"),
                "val_accuracy": val_metrics.get("accuracy"),
                "val_precision": val_metrics.get("precision"),
                "val_recall": val_metrics.get("recall"),
                "model_path": model_path,
            }
        )

    results_df = pd.DataFrame(results)
    results_path = os.path.join(signals_dir, "cart_meta_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"[6/6] CART meta-model training complete. Results saved to: {results_path}")


if __name__ == "__main__":
    main()

