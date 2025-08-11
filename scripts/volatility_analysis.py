
import pandas as pd
import numpy as np
import os
import sys

def calculate_volatility(df, lookback_periods):
    """Calculates rolling and EWMA volatility for multiple lookback periods."""
    print("Calculating daily returns...")
    # Group by instrument and calculate percentage change
    df['daily_return'] = df.groupby('instrument_token')['close'].pct_change()

    for period in lookback_periods:
        print(f"Calculating volatility for {period}-day lookback...")
        # Calculate rolling standard deviation of daily returns for each instrument
        rolling_std_col = f'volatility_rolling_std_{period}d'
        df[rolling_std_col] = df.groupby('instrument_token')['daily_return'].transform(
            lambda x: x.rolling(window=period, min_periods=period).std()
        )

        # Apply EWMA to the rolling standard deviation for smoothing
        ewma_vol_col = f'volatility_ewma_{period}d'
        df[ewma_vol_col] = df.groupby('instrument_token')[rolling_std_col].transform(
            lambda x: x.ewm(span=period, adjust=False).mean()
        )

    return df

def main():
    """Main function to run the volatility analysis."""
    input_path = 'data/processed/test_raw.parquet'
    output_path = 'scripts/volatility_analysis.csv'
    lookback_periods = [10, 20, 30, 50]

    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}. Please ensure the file exists.")
        sys.exit(1)

    # Sort data to ensure correct rolling calculations
    df.sort_values(by=['instrument_token', 'timestamp'], inplace=True)

    volatility_df = calculate_volatility(df.copy(), lookback_periods)

    # Select columns for the final output and drop rows with NaNs from calculations
    output_cols = ['instrument_token', 'timestamp', 'close', 'daily_return'] + \
                  [f'volatility_ewma_{period}d' for period in lookback_periods]
    final_df = volatility_df[output_cols].dropna()

    print(f"Saving volatility analysis to {output_path}...")
    final_df.to_csv(output_path, index=False)
    print(f"Volatility analysis complete. Output saved to {output_path}")

if __name__ == '__main__':
    main() 