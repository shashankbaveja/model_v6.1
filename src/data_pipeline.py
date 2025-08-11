import yaml
import pandas as pd
import os
import sys
from datetime import datetime

# Add the root directory to the Python path to allow importing 'myKiteLib'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myKiteLib import kiteAPIs, system_initialization

def load_config(config_path='config/parameters.yml'):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

def fetch_and_clean_data(api_client, sys_details, config):
    """Fetches raw data from the database and performs initial cleaning."""
    print("Fetching data from the database...")
    data_config = config['data']
    tokenList = sys_details.run_query_limit(data_config['token_list_query'])
    # print(len(tokenList))
    tokenList = map(str, tokenList)
    tokenList = ', '.join(tokenList)
    df = api_client.extract_data_from_db(
        from_date=data_config['training_start_date'],
        to_date=data_config['test_end_date'],
        interval='day',
        instrument_token=tokenList
    )
    if df is None or df.empty:
        print("Error: No data returned from the database. Exiting.")
        sys.exit(1)
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by time
    df.sort_values(by='timestamp', inplace=True)
    
    # Drop any duplicate timestamps
    df.drop_duplicates(subset=['timestamp','instrument_token'], keep='first', inplace=True)
    
    # --- Minimum Daily Volume Filtering ---
    # print("Filtering instruments by minimum daily volume...")
    min_daily_volumes = df.groupby('instrument_token')['volume'].min()
    initial_instrument_count_min = df['instrument_token'].nunique()
    tokens_with_sufficient_min_volume = min_daily_volumes[min_daily_volumes > 500].index
    df = df[df['instrument_token'].isin(tokens_with_sufficient_min_volume)]
    final_instrument_count_min = df['instrument_token'].nunique()
    if initial_instrument_count_min > final_instrument_count_min:
        print(f"Filtered out {initial_instrument_count_min - final_instrument_count_min} instruments out of {initial_instrument_count_min} with a daily volume of 500 on at least one day.")


    # Identify instruments that have data for all trading days
    last_60_days = sorted(df['timestamp'].unique())[-60:]
    df_last_60_days = df[df['timestamp'].isin(last_60_days)]
    unique_trading_days = df_last_60_days['timestamp'].nunique()
    tokens_with_data = df_last_60_days.groupby('instrument_token')['timestamp'].nunique()
    tokens_to_keep = tokens_with_data[tokens_with_data >= (unique_trading_days - 2)].index
    initial_instrument_count = df['instrument_token'].nunique()
    df = df[df['instrument_token'].isin(tokens_to_keep)]
    final_instrument_count = df['instrument_token'].nunique()
    print(f"Filtered out {initial_instrument_count - final_instrument_count} instruments out of {initial_instrument_count} with data for all trading days.")

    # --- Dollar Volume Filtering ---
    # Use the same last_60_days period for dollar volume analysis
    df_last_60_days_volume = df[df['timestamp'].isin(last_60_days)].copy()
    df_last_60_days_volume['dollar_volume'] = df_last_60_days_volume['close'] * df_last_60_days_volume['volume']
    average_dollar_volume = df_last_60_days_volume.groupby('instrument_token')['dollar_volume'].mean()
    tokens_with_sufficient_volume = average_dollar_volume[average_dollar_volume >= 2000000].index
    initial_instrument_count_volume = df['instrument_token'].nunique()
    df = df[df['instrument_token'].isin(tokens_with_sufficient_volume)]
    final_instrument_count_volume = df['instrument_token'].nunique()
    print(f"Filtered out {initial_instrument_count_volume - final_instrument_count_volume} instruments with an average daily dollar volume of less than 2,000,000 over the last 60 days.")

    # --- Circuit Hit Filtering ---
    # Use the same last_60_days period for circuit hit analysis
    df_last_60_days_circuit = df[df['timestamp'].isin(last_60_days)].copy()
    df_copy = df.copy()

    # Identify circuit hits: open=high=close (upper) or open=low=close (lower)
    circuit_hit = (
        (df_last_60_days_circuit['open'] == df_last_60_days_circuit['high']) &
        (df_last_60_days_circuit['open'] == df_last_60_days_circuit['close'])
    ) | (
        (df_last_60_days_circuit['open'] == df_last_60_days_circuit['low']) &
        (df_last_60_days_circuit['open'] == df_last_60_days_circuit['close'])
    )

    circuit_hit_all = (
        (df_copy['open'] == df_copy['high']) &
        (df_copy['open'] == df_copy['close'])
    ) | (
        (df_copy['open'] == df_copy['low']) &
        (df_copy['open'] == df_copy['close'])
    )
    df_last_60_days_circuit['is_circuit'] = circuit_hit
    df_copy['is_circuit'] = circuit_hit_all

    # Count circuit days per instrument
    circuit_days_count = df_last_60_days_circuit.groupby('instrument_token')['is_circuit'].sum()
    circuit_days_count_all = df_copy.groupby('instrument_token')['is_circuit'].sum()

    # Filter out instruments with excessive circuit hits
    tokens_with_frequent_circuits = circuit_days_count[circuit_days_count > 3].index
    tokens_with_frequent_circuits_all = circuit_days_count_all[circuit_days_count_all > 20].index

    initial_instrument_count_circuit = df['instrument_token'].nunique()
    df = df[~df['instrument_token'].isin(tokens_with_frequent_circuits)]
    df = df[~df['instrument_token'].isin(tokens_with_frequent_circuits_all)]
    final_instrument_count_circuit = df['instrument_token'].nunique()

    if initial_instrument_count_circuit > final_instrument_count_circuit:
        print(f"Filtered out {initial_instrument_count_circuit - final_instrument_count_circuit} instruments that hit circuits on more than 4 of the last 60 trading days.")

    print(f"Remaining instruments: {df['instrument_token'].nunique()}")
    print("Data cleaning complete.")
    # print(len(df))
    return df

def split_data(df, config):
    """Splits the dataframe into training, validation, and test sets."""
    print("Splitting data into training, validation, and test sets...")
    data_config = config['data']
    
    # Convert config dates to datetime for comparison
    train_start = pd.to_datetime(data_config['training_start_date'])
    train_end = pd.to_datetime(data_config['training_end_date']).replace(hour=23, minute=59, second=59)
    
    validation_start = train_end + pd.Timedelta(seconds=1)
    validation_end = pd.to_datetime(data_config['validation_end_date']).replace(hour=23, minute=59, second=59)
    
    test_start = validation_end + pd.Timedelta(seconds=1)
    test_end = pd.to_datetime(data_config['test_end_date']).replace(hour=23, minute=59, second=59)
    
    # Create a date mask for splitting
    train_df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)]
    validation_df = df[(df['timestamp'] >= validation_start) & (df['timestamp'] <= validation_end)]
    test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)]
    
    print(f"Training set: {len(train_df)} rows")
    print(f"Validation set: {len(validation_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    return train_df, validation_df, test_df

def save_datasets(train_df, validation_df, test_df, output_dir='data/processed'):
    """Saves the datasets to the specified directory."""
    print(f"Saving datasets to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_parquet(os.path.join(output_dir, 'train_raw.parquet'))
    validation_df.to_parquet(os.path.join(output_dir, 'validation_raw.parquet'))
    test_df.to_parquet(os.path.join(output_dir, 'test_raw.parquet'))
    
    print("Datasets saved successfully.")

def main():
    """Main function to run the data ingestion pipeline."""
    print("--- Starting Data Ingestion Pipeline ---")
    
    config = load_config()

    
    api_client = kiteAPIs()
    sys_details = system_initialization()
    
    full_df = fetch_and_clean_data(api_client, sys_details, config)
    print(len(full_df))
    train_df, validation_df, test_df = split_data(full_df, config)
    
    save_datasets(train_df, validation_df, test_df)
    
    print("--- Data Ingestion Pipeline Finished ---")
if __name__ == "__main__":
    main()
