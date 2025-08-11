import pandas as pd
import os
import sys

def main():
    """
    This script evaluates generated signals by merging them with their corresponding
    ground truth targets.
    """
    print("--- Starting Signal Evaluation Script ---")

    # Define paths
    signals_path = 'data/signals/unified_signals.csv'
    features_path = 'data/processed/test_combined_with_patterns_features.parquet'
    output_dir = 'reports'
    output_path = os.path.join(output_dir, 'signal_evaluation.csv')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read the signals file
    print(f"Reading signals from: {signals_path}")
    try:
        signals_df = pd.read_csv(signals_path)
    except FileNotFoundError:
        print(f"ERROR: Signals file not found at '{signals_path}'.")
        print("Please ensure you have run the signal generation pipeline first.")
        sys.exit(1)

    # 2. Read the processed features file with targets
    print(f"Reading features and targets from: {features_path}")
    try:
        features_df = pd.read_parquet(features_path)
    except FileNotFoundError:
        print(f"ERROR: Features file not found at '{features_path}'.")
        print("Please ensure you have run the full data and feature pipeline first.")
        sys.exit(1)

    # Ensure timestamp columns are in datetime format for a reliable merge
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])

    # 3. Select target columns and merge with signals
    print("Joining signals with targets on 'instrument_token' and 'timestamp'...")
    target_cols = ['instrument_token', 'timestamp', 'target_up', 'target_down']
    
    # Check if necessary target columns exist in the features dataframe
    required_targets = ['target_up', 'target_down']
    if not all(col in features_df.columns for col in required_targets):
        print(f"ERROR: Required target columns ({', '.join(required_targets)}) not found in {features_path}.")
        sys.exit(1)

    # Use a 'left' merge to keep all signals and add targets where they exist
    merged_df = pd.merge(
        signals_df,
        features_df[target_cols],
        on=['instrument_token', 'timestamp'],
        how='left'
    )
    
    # Check the result of the merge
    na_count = merged_df['target_up'].isna().sum()
    if na_count > 0:
        print(f"Warning: {na_count} signals could not be matched with a target. This might be expected if the signal set is larger than the test set.")

    # 4. Save the new file
    print(f"Saving evaluated signals to: {output_path}")
    merged_df.to_csv(output_path, index=False)

    print(f"--- Signal Evaluation Finished. Output saved to {output_path} ---")


if __name__ == "__main__":
    main()
