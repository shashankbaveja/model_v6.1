import os
import sys
import argparse
import pandas as pd
import joblib

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import load_config


def generate_signals_v2(entry_model_path_1: str, entry_model_path_2: str, exit_model_path: str, config: dict) -> str:
    """
    Generate unified signals with two entry probabilities and one exit probability.

    Returns the output CSV path.
    """
    print("--- Starting Unified Signal Generation V2 ---")

    processed_dir = 'data/processed'
    signals_dir = 'data/signals'
    os.makedirs(signals_dir, exist_ok=True)

    # Strategy name derived from first entry model; only used for feature file discovery
    try:
        strategy = os.path.basename(entry_model_path_1).split('_')[0]
    except IndexError:
        print(f"Error: Could not determine strategy from model name: {entry_model_path_1}")
        sys.exit(1)

    print(f"Loading data for strategy: {strategy.upper()}")
    feature_data_path = os.path.join(processed_dir, f'test_{strategy}_with_patterns_features.parquet')
    try:
        feature_df = pd.read_parquet(feature_data_path)
        raw_test_data = pd.read_parquet(os.path.join(processed_dir, 'test_raw.parquet'))
        print("  Loaded feature data and raw test data.")
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Have you run the feature generation pipeline? Details: {e}")
        sys.exit(1)

    # Prepare X_test by dropping targets and identifiers
    X_test = feature_df.drop(columns=['target_up', 'target_down', 'instrument_token', 'timestamp'], errors='ignore')

    # Load models
    for p in [entry_model_path_1, entry_model_path_2, exit_model_path]:
        if not os.path.exists(p):
            print(f"Error: Model not found at {p}")
            sys.exit(1)

    print(f"  - Loading entry model 1: {entry_model_path_1}")
    entry_model_1 = joblib.load(entry_model_path_1)
    print(f"  - Loading entry model 2: {entry_model_path_2}")
    entry_model_2 = joblib.load(entry_model_path_2)
    print(f"  - Loading exit model: {exit_model_path}")
    exit_model = joblib.load(exit_model_path)

    # Predict probabilities
    print("Generating signal probabilities (v2)...")
    entry_pred_proba_1 = entry_model_1.predict_proba(X_test)[:, 1]
    entry_pred_proba_2 = entry_model_2.predict_proba(X_test)[:, 1]
    exit_pred_proba = exit_model.predict_proba(X_test)[:, 1]

    # Assemble output
    signals_df = feature_df[['instrument_token', 'timestamp']].copy()
    signals_df['entry_signal_prob_1'] = entry_pred_proba_1
    signals_df['entry_signal_prob_2'] = entry_pred_proba_2
    signals_df['exit_signal_prob'] = exit_pred_proba

    final_signal_df = pd.merge(raw_test_data, signals_df, on=['instrument_token', 'timestamp'], how='left')
    for col in ['entry_signal_prob_1', 'entry_signal_prob_2', 'exit_signal_prob']:
        final_signal_df[col] = final_signal_df[col].fillna(0)

    output_filename = 'unified_signals_v2.csv'
    output_path = os.path.join(signals_dir, output_filename)
    final_signal_df.to_csv(output_path, index=False)
    print(f"--- Unified signals (v2) saved to {output_path} ---")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate unified trade signals (v2) from two entry models and one exit model.")
    parser.add_argument('--entry-model-path-1', type=str, required=True, help='Path to entry model 1 .joblib')
    parser.add_argument('--entry-model-path-2', type=str, required=True, help='Path to entry model 2 .joblib')
    parser.add_argument('--exit-model-path', type=str, required=True, help='Path to the exit (DOWN) model .joblib file')
    args = parser.parse_args()

    config = load_config()
    generate_signals_v2(
        entry_model_path_1=args.entry_model_path_1,
        entry_model_path_2=args.entry_model_path_2,
        exit_model_path=args.exit_model_path,
        config=config,
    )


if __name__ == "__main__":
    main()

