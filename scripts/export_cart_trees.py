import os
import sys
import glob
import argparse
import shutil
from typing import List, Tuple
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz


def discover_cart_models(input_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(input_dir, "*.joblib")))


def load_feature_names_from_signals(signals_csv: str, expected_len: int | None) -> List[str]:
    try:
        df = pd.read_csv(signals_csv, nrows=1)  # header only sufficient
        cols = sorted([c for c in df.columns if c.startswith("pred_")])
        if expected_len is not None and len(cols) == expected_len:
            return cols
        # Fallback to generic names if mismatch
    except Exception:
        pass
    # Generic fallback names
    n = expected_len if expected_len is not None else 0
    return [f"f{i}" for i in range(n)]


def parse_cart_filename(filename: str) -> Tuple[int, float, float]:
    """Extract L, TP, SL from cart_meta_up_L{L}_TP{TP}_SL{SL}.joblib using regex."""
    base = os.path.basename(filename)
    name = base[:-7] if base.endswith('.joblib') else base
    m = re.match(r"^cart_meta_up_L(?P<L>\d+)_TP(?P<TP>\d+(?:\.\d+)?)_SL(?P<SL>\d+(?:\.\d+)?)$", name)
    if not m:
        return -1, float('nan'), float('nan')
    L = int(m.group('L'))
    TP = float(m.group('TP'))
    SL = float(m.group('SL'))
    return L, TP, SL


def export_tree_dot_png(model_path: str, output_dir: str, feature_names: List[str], verbose: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(model_path))[0]
    dot_path = os.path.join(output_dir, f"{base}.dot")
    png_path = os.path.join(output_dir, f"{base}.png")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Skipping {model_path}: failed to load ({e})")
        return

    # Ensure feature name length matches model
    n_features = getattr(model, "n_features_in_", None)
    if n_features is None:
        print(f"Skipping {model_path}: model has no n_features_in_ attribute")
        return
    if len(feature_names) != n_features:
        # Pad or truncate to match length
        if len(feature_names) < n_features:
            feature_names = feature_names + [f"f{i}" for i in range(len(feature_names), n_features)]
        else:
            feature_names = feature_names[:n_features]

    try:
        export_graphviz(
            model,
            out_file=dot_path,
            feature_names=feature_names,
            class_names=["0", "1"],
            filled=True,
            rounded=True,
            special_characters=True,
        )
        if verbose:
            print(f"Wrote DOT: {dot_path}")
    except Exception as e:
        print(f"Failed to write DOT for {model_path}: {e}")
        return

    # Try to render PNG using Graphviz 'dot' if available
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        if verbose:
            print("Graphviz 'dot' not found on PATH. Skipping PNG render.")
        return

    try:
        os.system(f"\"{dot_bin}\" -Tpng \"{dot_path}\" -o \"{png_path}\"")
        if verbose:
            print(f"Wrote PNG: {png_path}")
    except Exception as e:
        print(f"Failed to render PNG for {model_path}: {e}")


def compute_and_save_leaf_stats(
    model_path: str,
    signals_csv: str,
    output_dir: str,
    verbose: bool = False,
) -> None:
    """Compute per-leaf samples, positives and precision using signals CSV.

    - X features: all pred_* columns (sorted)
    - y column: inferred from CART filename as target_up_L{L}_TP{TP}_SL{SL}
    """
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Skipping stats for {model_path}: failed to load ({e})")
        return

    n_features = getattr(model, "n_features_in_", None)
    if n_features is None:
        print(f"Skipping stats for {model_path}: model has no n_features_in_")
        return

    try:
        df = pd.read_csv(signals_csv)
    except Exception as e:
        print(f"Skipping stats for {model_path}: failed to read signals CSV ({e})")
        return

    pred_cols = sorted([c for c in df.columns if c.startswith("pred_")])
    if not pred_cols:
        print(f"Skipping stats for {model_path}: no pred_* columns in signals CSV")
        return

    # Align feature count (pad/truncate if needed)
    X = df[pred_cols].copy()
    if X.shape[1] < n_features:
        for i in range(X.shape[1], n_features):
            X[f"f_pad_{i}"] = 0.0
    elif X.shape[1] > n_features:
        X = X.iloc[:, :n_features]

    L, TP, SL = parse_cart_filename(model_path)
    target_col = f"target_up_L{L}_TP{TP}_SL{SL}"
    if target_col not in df.columns:
        print(f"Warning: target column {target_col} not found for {os.path.basename(model_path)}; stats will be skipped.")
        return
    y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)

    # Drop rows with NA in features
    mask_valid = X.notna().all(axis=1) & y.notna()
    X_valid = X[mask_valid]
    y_valid = y[mask_valid]
    if X_valid.empty:
        print(f"Skipping stats for {model_path}: no valid rows after filtering.")
        return

    try:
        leaf_ids = model.apply(X_valid)
    except Exception as e:
        print(f"Skipping stats for {model_path}: apply() failed ({e})")
        return

    stats = (
        pd.DataFrame({"leaf_id": leaf_ids, "y": y_valid.values})
        .groupby("leaf_id")
        .agg(samples=("y", "size"), positives=("y", "sum"))
        .reset_index()
    )
    stats["precision_pct"] = np.where(stats["samples"] > 0, 100.0 * stats["positives"] / stats["samples"], np.nan)

    base = os.path.splitext(os.path.basename(model_path))[0]
    out_path = os.path.join(output_dir, f"{base}_leaf_stats.csv")
    try:
        stats.to_csv(out_path, index=False)
        if verbose:
            print(f"Wrote leaf stats: {out_path}")
    except Exception as e:
        print(f"Failed to write leaf stats for {model_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export CART models to DOT and PNG")
    parser.add_argument("--input-dir", default="models/cart_meta", help="Directory with CART .joblib files")
    parser.add_argument("--output-dir", default="models/cart_meta/trees", help="Directory to write .dot/.png")
    parser.add_argument(
        "--signals-csv",
        default="data/signals/all_model_signals.csv",
        help="CSV to infer feature names (pred_*) from; falls back to generic names if unavailable",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose progress")
    args = parser.parse_args()

    model_paths = discover_cart_models(args.input_dir)
    if not model_paths:
        print(f"No CART models found under {args.input_dir}")
        sys.exit(0)

    # Try to infer feature names from signals CSV using the first model's feature count
    first_model = joblib.load(model_paths[0])
    n_features = getattr(first_model, "n_features_in_", None)
    feature_names = load_feature_names_from_signals(args.signals_csv, n_features)

    if args.verbose:
        print(f"Found {len(model_paths)} CART models. Exporting to {args.output_dir} ...")

    for path in model_paths:
        export_tree_dot_png(path, args.output_dir, feature_names, verbose=args.verbose)
        compute_and_save_leaf_stats(path, args.signals_csv, args.output_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()

