import pandas as pd
import yaml
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from catboost import CatBoostClassifier
import numpy as np

# Add the root directory for local library import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import load_config
from src.utils.modeling_utils import tune_hyperparameters


def generate_binary_targets(df, config):
    """
    Generates binary target variables based on the method specified in the config.
    The input dataframe 'df' is expected to have a multi-index of (instrument_token, timestamp).
    """
    print("Generating binary target variables...")
    target_config = config['target_generation']

    # Guard against empty dataframes from the filtering step
    if df.empty:
        print("  WARNING: Input dataframe for target generation is empty. Returning empty result.")
        df['target_up'] = pd.Series(dtype='int')
        df['target_down'] = pd.Series(dtype='int')
        return df

    # Sort index to ensure correct shifting within groups
    df.sort_index(inplace=True)

    target_method = target_config.get('method', 'simple')  # Default to simple if not specified
    print(f"Using '{target_method}' target generation method.")

    if target_method == 'volatility':
        lookahead = target_config['lookahead_candles']
        tp_multiplier = target_config['volatility_tp_multipler']
        sl_multiplier = target_config['volatility_sl_multipler']

        print(f"  - Lookahead: {lookahead} candles")
        print(f"  - TP Multiplier: {tp_multiplier}")
        print(f"  - SL Multiplier: {sl_multiplier}")

        def _apply_volatility_targets(group):
            """
            Calculates mutually exclusive targets based on the first event (TP or SL) hit.
            """
            profit_target_level = group['close'] + (group['close'] * group['volatility_ewma_30d'] * tp_multiplier)
            stop_loss_level = group['close'] - (group['close'] * group['volatility_ewma_30d'] * tp_multiplier)

            target_up = np.zeros(len(group), dtype=int)
            target_down = np.zeros(len(group), dtype=int)

            for i in range(len(group) - lookahead):
                window = group.iloc[i + 1: i + 1 + lookahead]

                # Find the index of the first hit for TP and SL
                tp_hits = window.index[window['close'] >= profit_target_level.iloc[i]]
                sl_hits = window.index[window['close'] <= stop_loss_level.iloc[i]]

                first_tp_hit = tp_hits.min() if not tp_hits.empty else pd.NaT
                first_sl_hit = sl_hits.min() if not sl_hits.empty else pd.NaT

                # Determine which was hit first
                if pd.notna(first_tp_hit) and (pd.isna(first_sl_hit) or first_tp_hit < first_sl_hit):
                    target_up[i] = 1
                elif pd.notna(first_sl_hit) and (pd.isna(first_tp_hit) or first_sl_hit <= first_tp_hit):
                    target_down[i] = 1

            group['target_up'] = target_up
            group['target_down'] = target_down
            return group

        df = df.groupby(level='instrument_token', group_keys=False).apply(_apply_volatility_targets)
        print("Volatility-based binary target variables generated.")

    else:  # Default to the original simple method
        lookahead_periods = target_config['lookahead_candles']
        threshold = target_config['threshold_percent'] / 100.0

        print(f"  - Lookahead: {lookahead_periods} candles")
        print(f"  - Threshold: {threshold * 100}%")

        def _apply_simple_targets(group):
            # Find the highest high and lowest low in the future lookahead window
            future_high = group['high'].rolling(window=lookahead_periods).max().shift(-lookahead_periods)
            future_low = group['low'].rolling(window=lookahead_periods).min().shift(-lookahead_periods)

            # Calculate the potential return to the highest high and lowest low
            up_return = (future_high - group['close']) / group['close']
            down_return = (future_low - group['close']) / group['close']

            # Set target if the threshold is ever met within the window
            group['target_up'] = (up_return >= threshold).astype(int)
            group['target_down'] = (down_return <= -threshold).astype(int)
            return group

        df = df.groupby(level='instrument_token', group_keys=False).apply(_apply_simple_targets)
        print("Simple binary target variables generated.")

    df.dropna(subset=['target_up', 'target_down'], inplace=True)
    return df


def recompute_targets_for_features(feature_df: pd.DataFrame, partition_name: str, config: dict) -> pd.DataFrame:
    """Recompute targets for the provided feature rows using raw OHLCV and current config.

    - Aligns to the exact rows in feature_df via an inner merge on ['instrument_token','timestamp']
    - Computes volatility_ewma_30d if needed
    - Calls generate_binary_targets (copied verbatim) to produce target_up/target_down
    - Returns a dataframe with columns: instrument_token, timestamp, target_up, target_down
    """
    raw_path = os.path.join('data', 'processed', f'{partition_name}_raw.parquet')
    try:
        raw_df = pd.read_parquet(raw_path)
    except FileNotFoundError as e:
        print(f"Error: Raw data file not found at {raw_path}. Details: {e}")
        sys.exit(1)

    # Ensure timestamp dtypes are aligned for a clean merge
    feature_rows = feature_df[['instrument_token', 'timestamp']].copy()
    feature_rows['timestamp'] = pd.to_datetime(feature_rows['timestamp'])
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

    # Keep only the OHLCV columns needed for target generation
    ohlcv_cols = ['instrument_token', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not set(ohlcv_cols).issubset(raw_df.columns):
        print("Error: Raw dataframe missing required OHLCV columns for target generation.")
        sys.exit(1)

    aligned_df = feature_rows.merge(raw_df[ohlcv_cols], on=['instrument_token', 'timestamp'], how='inner')

    # Compute minimal volatility feature if volatility-based targets are enabled
    target_method = config.get('target_generation', {}).get('method', 'simple')
    if target_method == 'volatility':
        aligned_df = aligned_df.sort_values(by=['instrument_token', 'timestamp']).copy()
        def _compute_volatility(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            group['daily_return'] = group['close'].pct_change()
            rolling_std = group['daily_return'].rolling(window=30, min_periods=30).std()
            group['volatility_ewma_30d'] = rolling_std.ewm(span=30, adjust=False).mean()
            return group
        aligned_df = aligned_df.groupby('instrument_token', group_keys=False).apply(_compute_volatility)

    # Set MultiIndex as expected by generate_binary_targets
    aligned_df.set_index(['instrument_token', 'timestamp'], inplace=True)
    aligned_with_targets = generate_binary_targets(aligned_df, config)

    # Return only identity + target columns for merging back
    aligned_with_targets = aligned_with_targets.reset_index()[
        ['instrument_token', 'timestamp', 'target_up', 'target_down']
    ]
    return aligned_with_targets

def load_data(strategy, target_name, config):
    """Loads feature and target data for a given strategy and interval."""
    print(f"Loading data for strategy: {strategy}, target: {target_name}")
    
    processed_dir = 'data/processed'
    
    base_train_filename = f'train_{strategy}_with_patterns_features.parquet'
    train_filename = os.path.join(processed_dir, base_train_filename)
    print(f"Train filename: {train_filename}")

    base_val_filename = f'validation_{strategy}_with_patterns_features.parquet'
    val_filename = os.path.join(processed_dir, base_val_filename)
    
    try:
        train_df = pd.read_parquet(train_filename)
        val_df = pd.read_parquet(val_filename)

        # Drop any pre-existing target columns in the feature files
        train_df = train_df.drop(columns=['target_up', 'target_down'], errors='ignore')
        val_df = val_df.drop(columns=['target_up', 'target_down'], errors='ignore')

        # Recompute targets freshly based on current config
        print("Recomputing targets for training partition...")
        train_targets = recompute_targets_for_features(train_df, partition_name='train', config=config)
        print("Recomputing targets for validation partition...")
        val_targets = recompute_targets_for_features(val_df, partition_name='validation', config=config)

        # Merge targets back into the featureframes
        train_df = train_df.merge(train_targets, on=['instrument_token', 'timestamp'], how='inner')
        val_df = val_df.merge(val_targets, on=['instrument_token', 'timestamp'], how='inner')

        # Build y and penalty from recomputed targets
        y_train = train_df.pop(target_name)
        y_val = val_df.pop(target_name)
        penalty_target_name = 'target_down' if target_name == 'target_up' else 'target_up'
        train_penalty = train_df[penalty_target_name]
        val_penalty = val_df[penalty_target_name]

        # Build feature matrices
        X_train = train_df.drop(columns=['target_up', 'target_down', 'instrument_token', 'timestamp'], errors='ignore')
        X_val = val_df.drop(columns=['target_up', 'target_down', 'instrument_token', 'timestamp'], errors='ignore')

    except FileNotFoundError as e:
        print(f"Error: Data file not found. Have you run feature generation and merging? Details: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Target column not found. Details: {e}")
        sys.exit(1)

    return X_train, y_train, X_val, y_val, train_penalty, val_penalty


def train_model(strategy, target_name, model_type, config):
    """Trains a single model for a given strategy, target, and model type."""
    X_train, y_train, X_val, y_val, train_penalty, val_penalty = load_data(strategy, target_name, config)
    
    print(f"Training a {model_type} model...")
    print(f"  Training data loaded. Number of features: {len(X_train.columns)}")

    # Tune hyperparameters
    best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, train_penalty, config)

    # Separate penalty_weight from other hyperparameters
    penalty_weight = best_params.pop('penalty_weight', 6.0) # Default to 6.0 if not found

    # Combine training and validation data for the final model
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    penalty_combined = pd.concat([train_penalty, val_penalty], ignore_index=True)
    
    # --- BUG FIX: Calculate scale_pos_weight for the combined dataset ---
    scale_pos_weight_combined = (y_combined == 0).sum() / (y_combined == 1).sum() if (y_combined == 1).sum() > 0 else 1

    # Create and train the final model with the best parameters
    final_model = CatBoostClassifier(
        random_state=config['modeling']['random_state'],
        verbose=0,
        scale_pos_weight=scale_pos_weight_combined,
        **best_params
    )
    
    # Apply penalty weights
    sample_weight = np.where(penalty_combined == 1, penalty_weight, 1)

    print("Fitting the final model on combined data...")
    final_model.fit(X_combined, y_combined, sample_weight=sample_weight)
    
    print("Model training complete.")

    # Save the model
    direction = 'up' if 'up' in target_name else 'down'
    
    # --- MODIFICATION: Create descriptive filename ---
    target_config = config.get('target_generation', {})
    l = target_config.get('lookahead_candles', 'N/A')
    tp = target_config.get('volatility_tp_multipler', 'N/A')
    sl = target_config.get('volatility_sl_multipler', 'N/A')
    
    model_filename = f'{strategy}_{direction}_{model_type}_L{l}_TP{tp}_SL{sl}_model.joblib'
    # --- End of modification ---

    model_path = os.path.join('models', model_filename)
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"Model saved successfully to {model_path}")

def main():
    """Main function to run the model training pipeline."""
    print("--- Starting Model Training Pipeline ---")
    config = load_config('config/parameters.yml')
    
    # Get parameters from the unified config
    model_config = config.get('modeling', {})
    model_types = model_config.get('model_types', ['lightgbm'])
    strategies = model_config.get('strategies_to_train', ['momentum', 'reversion','combined'])
    targets = model_config.get('targets', ['target_up', 'target_down'])
    
    for strategy in strategies:
        for model_type in model_types:
            print(f"\n--- Training {model_type.upper()} Models for {strategy.upper()} Strategy ---")
            for target_name in targets:
                train_model(strategy, target_name, model_type, config)

    print("\n--- Model Training Pipeline Finished ---")

if __name__ == "__main__":
    main()
