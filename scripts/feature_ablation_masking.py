import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from typing import List, Tuple, Dict

# Ensure project root is on path to reuse utilities if needed
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

ENTRY_MODEL_PATH = 'models/combined_up_catboost_L10_TP4.0_SL2.0_model.joblib'
EXIT_MODEL_PATH = 'models/combined_down_catboost_L15_TP4.0_SL2.0_model.joblib'

# Data locations (combined strategy, merged with patterns)
TRAIN_FEATURES_PATH = 'data/processed/train_combined_with_patterns_features.parquet'
VAL_FEATURES_PATH = 'data/processed/validation_combined_with_patterns_features.parquet'

# Thresholds
ENTRY_THRESHOLD = 0.70
EXIT_THRESHOLD = 0.35

OUTPUT_CSV_PATH = 'reports/ablation_masking_combined.csv'
OUTPUT_DROP_ORDER_JSON = 'reports/ablation_masking_drop_orders.json'


def _build_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].astype(int).copy()
    X = df.drop(columns=['instrument_token', 'timestamp', 'target_up', 'target_down'], errors='ignore').copy()
    # Ensure numeric and fill NAs (flags are expected to be ints); masking assumes zeros are inert
    for c in X.columns:
        if X[c].dtype.kind not in ('i', 'u', 'f', 'b'):  # non-numeric
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0)
    return X, y


def _align_columns(reference_cols: List[str], X: pd.DataFrame) -> pd.DataFrame:
    # Reindex to reference columns, fill missing with 0, drop extras if any
    out = X.reindex(columns=reference_cols, fill_value=0)
    return out


def _evaluate_predictions(y_true: pd.Series, y_pred_labels: np.ndarray) -> Dict[str, float]:
    total_predictions = int((y_pred_labels == 1).sum())
    coverage = float(total_predictions) / float(len(y_true)) if len(y_true) > 0 else 0.0

    tp = int(((y_true == 1) & (y_pred_labels == 1)).sum())
    fp = int(((y_true == 0) & (y_pred_labels == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fp_pct = 1.0 - precision

    return {
        'total_predictions': total_predictions,
        'coverage': coverage,
        'precision': precision,
        'recall': recall,
        'fp_pct': fp_pct,
    }


def _compute_drop_order(model, X_val: pd.DataFrame, y_val: pd.Series, *, model_path: str) -> List[str]:
    """
    Determine drop order using, in priority:
    1) Precomputed CSV from reports/feature_importance (FeatureImportance/gain)
    2) Model FeatureImportance (gain) without data
    3) PredictionValuesChange on validation pool

    Always return order of least-important -> most-important for masking.
    Only include features present in X_val.
    """
    # 1) Try precomputed CSV
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    csv_path = os.path.join('reports', 'feature_importance', f"{base_name}_feature_importance.csv")
    feature_cols = list(X_val.columns)
    if os.path.exists(csv_path):
        try:
            fi_df = pd.read_csv(csv_path)
            # Expect columns: feature, importance
            fi_df = fi_df[['feature', 'importance']]
            fi_df = fi_df.dropna(subset=['feature'])
            # Filter to present columns and sort ascending (least important first)
            fi_df = fi_df[fi_df['feature'].isin(feature_cols)].copy()
            fi_df.sort_values('importance', ascending=True, inplace=True)
            return list(fi_df['feature'].tolist())
        except Exception:
            pass

    # 2) Try model FeatureImportance (gain) without Pool
    try:
        importances = model.get_feature_importance(type='FeatureImportance')
        model_feature_names = getattr(model, 'feature_names_', None)
        if model_feature_names is None or len(model_feature_names) != len(importances):
            # Fallback: map by position to X_val
            imp_series = pd.Series(importances, index=feature_cols[:len(importances)])
        else:
            imp_series = pd.Series(importances, index=model_feature_names)
        imp_series = imp_series.reindex(feature_cols).dropna()
        return list(imp_series.sort_values(ascending=True).index)
    except Exception:
        pass

    # 3) Fallback to PredictionValuesChange with validation pool
    try:
        from catboost import Pool
        pool = Pool(X_val, y_val, feature_names=list(X_val.columns))
        importances = model.get_feature_importance(pool, type='PredictionValuesChange')
        imp_series = pd.Series(importances, index=X_val.columns)
        return list(imp_series.sort_values(ascending=True).index)
    except Exception:
        raise RuntimeError('Failed to compute feature importance for drop order')


def _predict_with_threshold(model, X: pd.DataFrame, threshold: float) -> np.ndarray:
    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= threshold).astype(int)
    return labels


def _mask_columns_zero(X: pd.DataFrame, cols_to_mask: List[str]) -> pd.DataFrame:
    if not cols_to_mask:
        return X.copy()
    X_masked = X.copy()
    cols_present = [c for c in cols_to_mask if c in X_masked.columns]
    if cols_present:
        X_masked[cols_present] = 0
    return X_masked


def run_ablation_for_model(
    model_path: str,
    target_col: str,
    threshold: float,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_label: str,
) -> Tuple[pd.DataFrame, List[str]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)

    X_train_raw, y_train = _build_X_y(train_df, target_col)
    X_val_raw, y_val = _build_X_y(val_df, target_col)

    # Align feature columns to validation order (ranking is computed on validation)
    feature_cols = list(X_val_raw.columns)
    X_train = _align_columns(feature_cols, X_train_raw)
    X_val = X_val_raw.copy()

    # Sanity: models expect fixed number of features; ensure shapes are identical to training
    # We assume the saved featureframes match training feature layout

    drop_order = _compute_drop_order(model, X_val, y_val, model_path=model_path)

    results_rows: List[Dict] = []

    num_features = len(feature_cols)
    for i in range(num_features):
        cols_to_mask = drop_order[: i + 1]
        remaining = num_features - (i + 1)
        dropped_feature = drop_order[i]

        # Train split
        X_train_masked = _mask_columns_zero(X_train, cols_to_mask)
        y_train_pred = _predict_with_threshold(model, X_train_masked, threshold)
        train_metrics = _evaluate_predictions(y_train, y_train_pred)

        results_rows.append({
            'model': model_label,
            'iter': i,
            'num_features_left': remaining,
            'dropped_feature': dropped_feature,
            'split': 'train',
            **train_metrics,
        })

        # Validation split
        X_val_masked = _mask_columns_zero(X_val, cols_to_mask)
        y_val_pred = _predict_with_threshold(model, X_val_masked, threshold)
        val_metrics = _evaluate_predictions(y_val, y_val_pred)

        results_rows.append({
            'model': model_label,
            'iter': i,
            'num_features_left': remaining,
            'dropped_feature': dropped_feature,
            'split': 'validation',
            **val_metrics,
        })

    results_df = pd.DataFrame(results_rows)
    return results_df, drop_order


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    # Load featureframes
    if not os.path.exists(TRAIN_FEATURES_PATH):
        raise FileNotFoundError(f"Missing train features at {TRAIN_FEATURES_PATH}")
    if not os.path.exists(VAL_FEATURES_PATH):
        raise FileNotFoundError(f"Missing validation features at {VAL_FEATURES_PATH}")

    train_df = pd.read_parquet(TRAIN_FEATURES_PATH)
    val_df = pd.read_parquet(VAL_FEATURES_PATH)

    # Entry model (target_up)
    entry_results, entry_drop_order = run_ablation_for_model(
        model_path=ENTRY_MODEL_PATH,
        target_col='target_up',
        threshold=ENTRY_THRESHOLD,
        train_df=train_df,
        val_df=val_df,
        model_label='entry',
    )

    # Exit model (target_down)
    exit_results, exit_drop_order = run_ablation_for_model(
        model_path=EXIT_MODEL_PATH,
        target_col='target_down',
        threshold=EXIT_THRESHOLD,
        train_df=train_df,
        val_df=val_df,
        model_label='exit',
    )

    all_results = pd.concat([entry_results, exit_results], ignore_index=True)
    all_results.to_csv(OUTPUT_CSV_PATH, index=False)
    
    # Save drop orders for reproducibility
    drop_orders = {
        'entry': entry_drop_order,
        'exit': exit_drop_order,
    }
    with open(OUTPUT_DROP_ORDER_JSON, 'w') as f:
        json.dump(drop_orders, f, indent=2)

    print(f"Ablation results written to {OUTPUT_CSV_PATH}")
    print(f"Drop orders saved to {OUTPUT_DROP_ORDER_JSON}")


if __name__ == '__main__':
    main()

