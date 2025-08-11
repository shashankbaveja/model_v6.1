import pandas as pd
import numpy as np
import yaml
import os
import sys
import re
import warnings
# Suppress the specific FutureWarning from the 'ta' library
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import the 'ta' library for technical analysis
import ta
from ta.utils import dropna

# Add the root directory for local library import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_pipeline import load_config


def parse_feature_strategies(file_path='feature_list.md'):
    """Parses the feature list from the markdown file into strategy groups."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Feature list file not found at {file_path}")
        sys.exit(1)

    strategies = {
        "momentum": [],
        "reversion": [],
        "common": []
    }
    current_strategy = None

    for line in lines:
        line_stripped = line.strip()
        if "## Momentum Features" in line_stripped:
            current_strategy = "momentum"
        elif "## Mean Reversion Features" in line_stripped:
            current_strategy = "reversion"
        elif "## Common Features" in line_stripped:
            current_strategy = "common"
        elif "## To be integrated in future" in line_stripped:
            current_strategy = None
        
        # Ensure it's a valid feature line: starts with '*' and has a description
        if current_strategy and line_stripped.startswith('* ') and ':' in line_stripped:
            feature_name = line_stripped.split(':')[0].replace('*', '').strip()
            strategies[current_strategy].append(feature_name)
            
    return strategies['momentum'], strategies['reversion'], strategies['common']


def calculate_daily_vwap(df):
    """Calculates a daily resetting VWAP."""
    df['date_day'] = df.index.date
    df['tpv'] = df['volume'] * (df['high'] + df['low'] + df['close']) / 3
    df['cum_tpv'] = df.groupby('date_day')['tpv'].cumsum()
    df['cum_volume'] = df.groupby('date_day')['volume'].cumsum()
    df['VWAP_D'] = df['cum_tpv'] / df['cum_volume']
    df.drop(columns=['date_day', 'tpv', 'cum_tpv', 'cum_volume'], inplace=True)
    return df


def add_base_indicators(df):
    """Adds all necessary base TA indicators to a dataframe."""
    df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    df = calculate_daily_vwap(df)
    return df


def calculate_all_features(df):
    """
    Calculates all defined binary features on the given dataframe.
    This function now supports multi-instrument data by grouping operations
    by 'instrument_token'.
    The dataframe is expected to have 'instrument_token' and 'timestamp' columns.
    """
    print(f"Starting feature calculation for {len(df)} rows across {df['instrument_token'].nunique()} instruments...")

    def _calculate_features_for_group(group_df):
        """Helper function to apply all feature calculations to a single-instrument group."""
        
        group_df = group_df.copy()
        if 'timestamp' in group_df.columns:
            group_df.set_index('timestamp', inplace=True)
        
        # This function adds all standard 'ta' features.
        group_df = add_base_indicators(group_df)
        
        # Add any specific indicators not included in the standard library run
        group_df['sma_8'] = ta.trend.sma_indicator(group_df['close'], window=8, fillna=True)
        group_df['sma_21'] = ta.trend.sma_indicator(group_df['close'], window=21, fillna=True)
        # group_df['sma_100'] = ta.trend.sma_indicator(group_df['close'], window=200, fillna=True)
        group_df['sma_200'] = ta.trend.sma_indicator(group_df['close'], window=200, fillna=True)

        # --- Volatility Calculation for Dynamic Targets ---
        group_df['daily_return'] = group_df['close'].pct_change()
        rolling_std = group_df['daily_return'].rolling(window=30, min_periods=30).std()
        group_df['volatility_ewma_30d'] = rolling_std.ewm(span=30, adjust=False).mean()
        # --- End Volatility Calculation ---

        # --- Phase 1: Add custom base indicators for new features ---
        group_df['kama_14'] = ta.momentum.kama(group_df['close'], window=14, fillna=True)
        group_df['donchian_hband_10'] = ta.volatility.donchian_channel_hband(group_df['high'], group_df['low'], group_df['close'], window=10, fillna=True)
        group_df['cmf_10'] = ta.volume.chaikin_money_flow(group_df['high'], group_df['low'], group_df['close'], group_df['volume'], window=10, fillna=True)
        
        # Calculate True Range for skewness
        tr1 = group_df['high'] - group_df['low']
        tr2 = abs(group_df['high'] - group_df['close'].shift(1))
        tr3 = abs(group_df['low'] - group_df['close'].shift(1))
        group_df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # --- End of Phase 1 ---

        feature_df = pd.DataFrame(index=group_df.index)

        momentum_f, reversion_f, common_f = parse_feature_strategies()
        features_to_build = list(set(momentum_f + reversion_f + common_f))
        features_to_build.append('market_regime_up')
        # features_to_build.append('market_regime_down')

                # --- Add the 13 new features to the list to be built ---
        new_features = [
            'flag_mom5_pos', 'flag_pvt_slope', 'flag_macd_pos', 'flag_ema_xover',
            'flag_sma20_slope', 'flag_kama_diff', 'flag_donch_up', 'flag_ch_vol',
            'flag_tr_skew', 'flag_obv_slope', 'flag_vol_roc', 'flag_cmf_pos',
            'flag_vpt_div', 'flag_vwap_dev'
        ]
        features_to_build.extend(new_features)
        # --- End of new feature addition ---
        
        for feature in features_to_build:
            try:
                # --- Momentum Features (Calculated on the resampled df) ---
                if feature == 'MA_5_above_MA_13_1min':
                    feature_df[feature] = (ta.trend.sma_indicator(group_df['close'], window=5, fillna=True) > ta.trend.sma_indicator(group_df['close'], window=13, fillna=True)).astype(int)
                elif feature == 'MA_8_above_MA_21_5min':
                    feature_df[feature] = (group_df['sma_8'] > group_df['sma_21']).astype(int)
                elif feature == 'Price_above_MA_20_5min':
                    feature_df[feature] = (group_df['close'] > group_df['trend_sma_slow']).astype(int) # trend_sma_slow is 20-period
                elif feature == 'EMA_12_above_EMA_26_5min':
                    feature_df[feature] = (group_df['trend_ema_fast'] > group_df['trend_ema_slow']).astype(int)
                elif feature == 'Price_above_VWAP_5min':
                    feature_df[feature] = (group_df['close'] > group_df['VWAP_D']).astype(int)
                elif feature == 'VWAP_above_MA_20_5min':
                    feature_df[feature] = (group_df['VWAP_D'] > group_df['trend_sma_slow']).astype(int)
                elif feature == 'Price_velocity_positive_5min':
                    feature_df[feature] = (group_df['close'].diff(5) > 0).astype(int)
                elif feature == 'Price_acceleration_positive_5min':
                    feature_df[feature] = (group_df['close'].diff(5).diff(5) > 0).astype(int)
                elif feature == 'RSI_above_50_5min':
                    feature_df[feature] = (group_df['momentum_rsi'] > 50).astype(int)
                elif feature == 'Stoch_K_above_D_5min':
                    feature_df[feature] = (group_df['momentum_stoch'] > group_df['momentum_stoch_signal']).astype(int)
                elif feature == 'MACD_bullish_crossover_5min':
                    hist = group_df['trend_macd_diff']
                    feature_df[feature] = ((hist > 0) & (hist.shift(1) < 0)).astype(int)
                elif feature == 'MACD_bearish_crossover_5min':
                    hist = group_df['trend_macd_diff']
                    feature_df[feature] = ((hist < 0) & (hist.shift(1) > 0)).astype(int)
                elif feature == 'MACD_histogram_positive':
                    feature_df[feature] = (group_df['trend_macd_diff'] > 0).astype(int)
                elif feature == 'MACD_above_zero_line':
                    feature_df[feature] = (group_df['trend_macd'] > 0).astype(int)
                elif feature == 'ROC_positive_5min':
                    feature_df[feature] = (group_df['momentum_roc'] > 0).astype(int)
                elif feature == 'Plus_DI_above_Minus_DI':
                    feature_df[feature] = (group_df['trend_adx_pos'] > group_df['trend_adx_neg']).astype(int)
                elif feature == 'ADX_rising_5bars':
                    feature_df[feature] = (group_df['trend_adx'].diff(5) > 0).astype(int)
                elif feature == 'CCI_zero_line_cross_up':
                    feature_df[feature] = ((group_df['trend_cci'] > 0) & (group_df['trend_cci'].shift(1) < 0)).astype(int)
                elif feature == 'High_above_prev_high_3bars':
                    feature_df[feature] = (group_df['high'] > group_df['high'].rolling(3).max().shift(1)).astype(int)
                elif feature == 'Price_move_above_1p5_ATR':
                    feature_df[feature] = (abs(group_df['close'].diff()) > (1.5 * group_df['volatility_atr'])).astype(int)
                
                # --- Mean Reversion Features (Calculated on the resampled df) ---
                elif feature == 'RSI_overbought_70_5min':
                    feature_df[feature] = (group_df['momentum_rsi'] > 70).astype(int)
                elif feature == 'RSI_oversold_30_5min':
                    feature_df[feature] = (group_df['momentum_rsi'] < 30).astype(int)
                elif feature == 'Stoch_overbought_80_5min':
                    feature_df[feature] = (group_df['momentum_stoch'] > 80).astype(int)
                elif feature == 'Stoch_oversold_20_5min':
                    feature_df[feature] = (group_df['momentum_stoch'] < 20).astype(int)
                elif feature == 'Price_above_BB_upper_5min':
                    feature_df[feature] = (group_df['close'] > group_df['volatility_bbh']).astype(int)
                elif feature == 'Price_below_BB_lower_5min':
                    feature_df[feature] = (group_df['close'] < group_df['volatility_bbl']).astype(int)
                elif feature == 'WilliamsR_overbought_minus20':
                    feature_df[feature] = (group_df['momentum_wr'] > -20).astype(int)
                elif feature == 'CCI_overbought_above_100':
                    feature_df[feature] = (group_df['trend_cci'] > 100).astype(int)
                elif feature == 'CCI_oversold_below_minus100':
                    feature_df[feature] = (group_df['trend_cci'] < -100).astype(int)
                
                # --- New Features (Phase 2) ---
                elif feature == 'flag_mom5_pos':
                    feature_df[feature] = (group_df['close'].diff(5) > 0).astype(int)
                elif feature == 'flag_pvt_slope':
                    feature_df[feature] = (group_df['volume_vpt'].diff(5) > 0).astype(int) # PVT is same as VPT
                elif feature == 'flag_macd_pos':
                    feature_df[feature] = (group_df['trend_macd_diff'] > 0).astype(int)
                elif feature == 'flag_ema_xover':
                    feature_df[feature] = (group_df['trend_ema_fast'] > group_df['trend_ema_slow']).astype(int)
                elif feature == 'flag_sma20_slope':
                    feature_df[feature] = (group_df['trend_sma_slow'].diff(5) > 0).astype(int) # trend_sma_slow is 20-period
                elif feature == 'flag_kama_diff':
                    feature_df[feature] = ((group_df['kama_14'] - group_df['close']) / group_df['close'] > 0).astype(int)
                elif feature == 'flag_donch_up':
                    feature_df[feature] = (group_df['close'] > group_df['donchian_hband_10']).astype(int)
                elif feature == 'flag_ch_vol': # This is ATR Volatility Spike, not Chaikin Volatility
                    atr = group_df['volatility_atr']
                    feature_df[feature] = ((atr.diff(5) / atr.shift(5)) > 0.25).astype(int)
                elif feature == 'flag_tr_skew':
                    feature_df[feature] = (group_df['true_range'].rolling(14).skew() > 0).astype(int)
                elif feature == 'flag_obv_slope':
                    feature_df[feature] = (group_df['volume_obv'].diff(5) > 0).astype(int)
                elif feature == 'flag_vol_roc':
                    feature_df[feature] = ((group_df['volume'].diff(5) / group_df['volume'].shift(5)) > 0.20).astype(int)
                elif feature == 'flag_cmf_pos':
                    feature_df[feature] = (group_df['cmf_10'] > 0).astype(int)
                elif feature == 'flag_vpt_div':
                    price_up = group_df['close'].diff(1) > 0
                    vpt_down = group_df['volume_vpt'].diff(1) < 0
                    price_down = group_df['close'].diff(1) < 0
                    vpt_up = group_df['volume_vpt'].diff(1) > 0
                    
                    # Default to 0
                    feature_df[feature] = 0
                    # Bearish divergence
                    feature_df.loc[price_up & vpt_down, feature] = -1
                    # Bullish divergence
                    feature_df.loc[price_down & vpt_up, feature] = 1
                elif feature == 'flag_vwap_dev':
                    feature_df[feature] = (abs(group_df['close'] - group_df['VWAP_D']) / group_df['VWAP_D'] > 0.01).astype(int)
                
                # --- Common Features (Calculated on the resampled df) ---
                elif feature == 'ADX_strong_trend_above_25':
                    feature_df[feature] = (group_df['trend_adx'] > 25).astype(int)
                elif feature == 'BB_squeeze_5min':
                    feature_df[feature] = (group_df['volatility_bbw'] < group_df['volatility_bbw'].rolling(20).mean()).astype(int)
                elif feature == 'BB_expansion_5min':
                    feature_df[feature] = (group_df['volatility_bbw'] > group_df['volatility_bbw'].rolling(20).mean()).astype(int)
                elif feature == 'High_ATR_5min':
                    feature_df[feature] = (group_df['volatility_atr'] > group_df['volatility_atr'].rolling(20).mean()).astype(int)
                elif feature == 'Low_ATR_5min':
                    feature_df[feature] = (group_df['volatility_atr'] < group_df['volatility_atr'].rolling(20).mean()).astype(int)
                elif feature == 'Volume_above_20MA_5min':
                    feature_df[feature] = (group_df['volume'] > group_df['volume'].rolling(20).mean()).astype(int)
                elif feature == 'Volume_increasing_3bars':
                    feature_df[feature] = ((group_df['volume'] > group_df['volume'].shift(1)) & (group_df['volume'].shift(1) > group_df['volume'].shift(2))).astype(int)
                elif feature == 'IS_MORNING_SESSION':
                    feature_df[feature] = ((group_df.index.time >= pd.to_datetime('09:15').time()) & (group_df.index.time < pd.to_datetime('11:30').time())).astype(int)
                elif feature == 'IS_LUNCH_SESSION':
                    feature_df[feature] = ((group_df.index.time >= pd.to_datetime('11:30').time()) & (group_df.index.time < pd.to_datetime('13:30').time())).astype(int)
                elif feature == 'IS_AFTERNOON_SESSION':
                    feature_df[feature] = ((group_df.index.time >= pd.to_datetime('13:30').time()) & (group_df.index.time <= pd.to_datetime('15:30').time())).astype(int)
                elif feature == 'DAY_OF_WEEK':
                    days = pd.get_dummies(group_df.index.day_name(), prefix='day').astype(int)
                    for day_col in days.columns:
                        feature_df[day_col] = days[day_col]
                elif feature == 'market_regime_up':
                    feature_df[feature] = (group_df['close'] > group_df['sma_200']).astype(int)
                # elif feature == 'market_regime_down':
                #     feature_df[feature] = (group_df['close'] < group_df['sma_200']).astype(int)

                # # --- Breakout/Reversal Level Features ---
                # # NOTE: The name 'minute' or 'day' in the feature string now refers to a number of CANDLES.
                # elif 'high' in feature or 'low' in feature:
                #     match = re.search(r'last_(\d+)_(minute|day)', feature)
                #     if match:
                #         val = int(match.group(1))
                #         period_type = match.group(2)
                        
                #         # Correctly calculate lookback in candles based on interval
                #         candles_per_day = 375 / interval
                #         period = val if period_type == 'minute' else int(val * candles_per_day)
                        
                #         threshold = 0.001
                #         if 'within' in feature:
                #             if 'high' in feature:
                #                 highs = group_df['high'].rolling(window=period).max()
                #                 feature_df[feature] = ((group_df['close'] <= highs) & (group_df['close'] >= highs * (1 - threshold))).astype(int)
                #             else: # low
                #                 lows = group_df['low'].rolling(window=period).min()
                #                 feature_df[feature] = ((group_df['close'] >= lows) & (group_df['close'] <= lows * (1 + threshold))).astype(int)
                #         elif 'above' in feature:
                #             feature_df[feature] = (group_df['close'] > group_df['high'].rolling(window=period).max() * (1 + threshold)).astype(int)
                #         elif 'below' in feature:
                #             feature_df[feature] = (group_df['close'] < group_df['low'].rolling(window=period).min() * (1 - threshold)).astype(int)

            except Exception as e:
                print(f"Could not build feature '{feature}'. Error: {e}")

        final_df = group_df[['open', 'high', 'low', 'close', 'volume', 'volatility_ewma_30d']].join(feature_df)
        return final_df

    # Sort by instrument and time to ensure correct group processing
    df = df.sort_values(by=['instrument_token', 'timestamp'])
    
    # Filter out instruments with insufficient data to prevent calculation errors
    MIN_DATAPOINTS = 30 # A safe buffer for max lookback periods (e.g., 200-period SMA)
    print(f"Filtering out instruments with fewer than {MIN_DATAPOINTS} data points...")
    original_instruments = df['instrument_token'].nunique()
    df = df.groupby('instrument_token').filter(lambda x: len(x) >= MIN_DATAPOINTS)
    # breakpoint()
    filtered_instruments = df['instrument_token'].nunique()
    print(f"Filtered instruments: {original_instruments} -> {filtered_instruments}")

    # Apply the feature calculation to each instrument group
    # The result will have a multi-index: (instrument_token, timestamp)
    # include_groups=False prevents the grouping key from being passed to the function,
    # resolving a DeprecationWarning and making behavior consistent.
    all_features_df = df.groupby('instrument_token').apply(_calculate_features_for_group, include_groups=False)
    
    # Drop initial NaNs from indicator calculations
    all_features_df.dropna(inplace=True)
    
    print(f"Feature calculation complete. Total features created: {len(all_features_df.columns) - 6}") # -6 for ohlcv + vol
    return all_features_df


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

    target_method = target_config.get('method', 'simple') # Default to simple if not specified
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
                window = group.iloc[i+1 : i+1+lookahead]
                
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

    else: # Default to the original simple method
        lookahead_periods = target_config['lookahead_candles']
        threshold = target_config['threshold_percent'] / 100.0
        
        print(f"  - Lookahead: {lookahead_periods} candles")
        print(f"  - Threshold: {threshold*100}%")

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
        
    

def main():
    """Main function to run the feature engineering pipeline."""
    print("--- Starting Feature Engineering Pipeline ---")
    
    config = load_config('config/parameters.yml')
    
    momentum_features, reversion_features, common_features = parse_feature_strategies()
    momentum_cols_base = list(set(momentum_features + common_features + ['market_regime_up']))
    reversion_cols_base = list(set(reversion_features + common_features + ['market_regime_up']))

    new_features_to_save = [
        'flag_mom5_pos', 'flag_pvt_slope', 'flag_macd_pos', 'flag_ema_xover',
        'flag_sma20_slope', 'flag_kama_diff', 'flag_donch_up', 'flag_ch_vol',
        'flag_tr_skew', 'flag_obv_slope', 'flag_vol_roc', 'flag_cmf_pos',
        'flag_vpt_div', 'flag_vwap_dev'
    ]
    # Add to both momentum and reversion so they are included in combined
    momentum_cols_base.extend(new_features_to_save)
    reversion_cols_base.extend(new_features_to_save)
    # --- End of fix ---

    
    # print(f"Momentum model will use {len(momentum_cols_base)} base features.")
    # print(f"Mean Reversion model will use {len(reversion_cols_base)} base features.")

    try:
        train_raw = pd.read_parquet('data/processed/train_raw.parquet')
        validation_raw = pd.read_parquet('data/processed/validation_raw.parquet')
        test_raw = pd.read_parquet('data/processed/test_raw.parquet')
    except FileNotFoundError as e:
        print(f"Error: Raw data file not found. Have you run the data pipeline first? Details: {e}")
        sys.exit(1)

    datasets = {"train": train_raw, "validation": validation_raw, "test": test_raw}

    
    
    output_dir = 'data/features'
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df_raw in datasets.items():
        print(f"\n--- Processing {name} dataset ---")
        
        
        df_resampled = df_raw.copy()

        
        all_features_df = calculate_all_features(df_resampled)
        df_with_targets = generate_binary_targets(all_features_df, config)
        
        # The dataframe now has a multi-index (instrument_token, timestamp)
        # We need to handle this when saving. Resetting index is easiest.
        df_with_targets.reset_index(inplace=True)

        target_cols = ['target_up', 'target_down']
        # Also add instrument_token to the columns to keep
        core_cols = ['instrument_token', 'timestamp'] + target_cols

        available_cols = df_with_targets.columns
        day_cols = [col for col in available_cols if col.startswith('day_')]
        
        def get_final_cols(base_cols):
            final_cols = [col for col in base_cols if col in available_cols or col == 'DAY_OF_WEEK']
            if 'DAY_OF_WEEK' in final_cols:
                final_cols.remove('DAY_OF_WEEK')
                final_cols.extend(day_cols)
            return list(set(final_cols))

        momentum_cols = get_final_cols(momentum_cols_base)
        reversion_cols = get_final_cols(reversion_cols_base)
        combined_cols = list(set(momentum_cols + reversion_cols))

        print(f"  Final momentum feature count for {name}: {len(momentum_cols)}")
        print(f"  Final reversion feature count for {name}: {len(reversion_cols)}")
        print(f"  Final combined feature count for {name}: {len(combined_cols)}")

        momentum_df = df_with_targets[momentum_cols + core_cols].copy()
        momentum_path = os.path.join(output_dir, f'{name}_momentum_features.parquet')
        momentum_df.to_parquet(momentum_path, index=False)
        print(f"Saved momentum features for {name} to {momentum_path}")

        reversion_df = df_with_targets[reversion_cols + core_cols].copy()
        reversion_path = os.path.join(output_dir, f'{name}_reversion_features.parquet')
        reversion_df.to_parquet(reversion_path, index=False)
        print(f"Saved mean reversion features for {name} to {reversion_path}")

        combined_df = df_with_targets[combined_cols + core_cols].copy()
        combined_path = os.path.join(output_dir, f'{name}_combined_features.parquet')
        combined_df.to_parquet(combined_path, index=False)
        print(f"Saved combined features for {name} to {combined_path}")

    print("\n--- Feature Engineering Pipeline Finished ---")

if __name__ == "__main__":
    main()
