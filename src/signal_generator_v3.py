import os
import sys
import pandas as pd
import numpy as np
import joblib

# When running as `python src/signal_generator_v3.py`, the `src` directory is on sys.path
from data_pipeline import load_config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from myKiteLib import system_initialization


def _load_parquet_or_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _infer_strategy_from_model_path(model_path: str) -> str:
    base = os.path.basename(model_path)
    try:
        return base.split('_')[0]
    except Exception:
        return 'combined'


def _compute_volatility_ewma_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values(['instrument_token', 'timestamp'])
    df['daily_return'] = df.groupby('instrument_token')['close'].pct_change()
    rolling_std = df.groupby('instrument_token')['daily_return'].transform(lambda x: x.rolling(window=30).std())
    df['volatility_ewma_30d'] = rolling_std.ewm(span=30, adjust=False).mean()
    return df[['instrument_token', 'timestamp', 'volatility_ewma_30d']]


def generate_unified_signals_v3_from_config(config: dict) -> tuple[str, str]:
    # Directories (default if not present)
    processed_dir = config.get('paths', {}).get('processed_dir', 'data/processed')
    signals_dir = config.get('paths', {}).get('signals_dir', 'data/signals')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(signals_dir, exist_ok=True)

    # Models from config
    bt_cfg = config.get('backtest_v3', {})
    entry_models = bt_cfg.get('entry_models', [])
    if len(entry_models) < 2:
        raise ValueError('Expected two entry models in config.backtest_v3.entry_models')
    entry_model_path_1 = entry_models[0].get('model_path')
    entry_model_path_2 = entry_models[1].get('model_path')
    exit_model_path = bt_cfg.get('exit_model', {}).get('model_path')
    if not entry_model_path_1 or not entry_model_path_2 or not exit_model_path:
        raise ValueError('Model paths missing in config.backtest_v3')

    strategy = _infer_strategy_from_model_path(entry_model_path_1)

    feature_path = os.path.join(processed_dir, f'test_{strategy}_with_patterns_features.parquet')
    raw_path = os.path.join(processed_dir, 'test_raw.parquet')

    feature_df = _load_parquet_or_csv(feature_path)
    raw_df = _load_parquet_or_csv(raw_path)

    # Prepare X_test excluding targets and identifiers
    X_test = feature_df.drop(columns=['target_up', 'target_down', 'instrument_token', 'timestamp'], errors='ignore')

    # Load models
    for p in [entry_model_path_1, entry_model_path_2, exit_model_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model not found at {p}")
    entry_model_1 = joblib.load(entry_model_path_1)
    entry_model_2 = joblib.load(entry_model_path_2)
    exit_model = joblib.load(exit_model_path)

    # Predict probabilities
    entry_pred_proba_1 = entry_model_1.predict_proba(X_test)[:, 1]
    entry_pred_proba_2 = entry_model_2.predict_proba(X_test)[:, 1]
    exit_pred_proba = exit_model.predict_proba(X_test)[:, 1]

    # Build signals dataframe
    signals_df = feature_df[['instrument_token', 'timestamp']].copy()
    signals_df['entry_signal_prob_1'] = entry_pred_proba_1
    signals_df['entry_signal_prob_2'] = entry_pred_proba_2
    signals_df['exit_signal_prob'] = exit_pred_proba

    # Load thresholds or use defaults mirroring current logic
    # Entry tiers
    tier_cfg = bt_cfg.get('entry_tiers', {})
    m6_p1 = tier_cfg.get('m6_p1_thr', 0.78)
    m6_p2 = tier_cfg.get('m6_p2_thr', 0.77)
    m4_p1 = tier_cfg.get('m4_p1_thr', 0.70)
    m4_p2 = tier_cfg.get('m4_p2_thr', 0.70)
    m1_p1 = tier_cfg.get('m1_p1_thr', 0.65)
    m1_p2 = tier_cfg.get('m1_p2_thr', 0.65)
    # Exit
    exit_model_cfg = bt_cfg.get('exit_model', {})
    exit_threshold_for_signal = exit_model_cfg.get('threshold_for_exit_signal', 0.35)
    threshold_for_entry_filter = exit_model_cfg.get('threshold_for_entry_filter', 0.25)

    # Tiering to entry_flag and capital_multiplier with exit entry filter
    m6 = (signals_df['entry_signal_prob_1'] >= m6_p1) & (signals_df['entry_signal_prob_2'] >= m6_p2)
    m4 = (signals_df['entry_signal_prob_1'] >= m4_p1) & (signals_df['entry_signal_prob_2'] >= m4_p2)
    exit_filter_ok = (signals_df['exit_signal_prob'] < threshold_for_entry_filter)

    signals_df['capital_multiplier'] = 0
    signals_df.loc[m4 & exit_filter_ok, 'capital_multiplier'] = 1
    signals_df.loc[m6 & exit_filter_ok, 'capital_multiplier'] = 2
    signals_df['entry_flag'] = (signals_df['capital_multiplier'] > 0).astype(int)

    # Exit flag based on exit threshold
    signals_df['exit_flag'] = (signals_df['exit_signal_prob'] > exit_threshold_for_signal).astype(int)

    # Compute volatility feature file for backtester consumption
    vol_df = _compute_volatility_ewma_from_raw(raw_df)

    # Save outputs
    sys_details = system_initialization()
    token_name_list = sys_details.run_query_full("Select distinct instrument_token, tradingsymbol from kiteconnect.instruments_zerodha")
    token_name = pd.DataFrame(token_name_list)
    token_name = token_name.drop_duplicates(subset=['instrument_token'])
    signals_df = signals_df.merge(token_name, on='instrument_token', how='left')


    unified_path = os.path.join(signals_dir, 'unified_signals_v3.csv')
    save_cols = ['instrument_token', 'tradingsymbol', 'timestamp', 'entry_flag', 'exit_flag', 'capital_multiplier']
    signals_df[save_cols].to_csv(unified_path, index=False)

    vol_features_path = os.path.join(processed_dir, 'backtest_features_with_vol_v3.parquet')
    vol_df.to_parquet(vol_features_path, index=False)

    return unified_path, vol_features_path


def main():
    config = load_config()
    unified_path, vol_features_path = generate_unified_signals_v3_from_config(config)
    print(f"Unified signals saved to: {unified_path}")
    print(f"Volatility features saved to: {vol_features_path}")


if __name__ == '__main__':
    main()

