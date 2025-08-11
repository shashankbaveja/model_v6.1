import os
import sys
import subprocess
from datetime import datetime
import time
import pandas as pd
import json
from kiteconnect.exceptions import KiteException, InputException
import google.generativeai as genai

# --- Configuration & Imports ---
LOG_DIR = 'logs'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myKiteLib import OrderPlacement, kiteAPIs, system_initialization
from src.live_trading_helpers import (
    load_config, get_holdings, run_gemini_bridge
)

# --- Environment Setup ---
from dotenv import load_dotenv
load_dotenv()
try:
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    print("Please make sure your .env file is configured correctly with GEMINI_API_KEY.")
    exit()

def run_step(command, log_file):
    """Executes a command as a subprocess and logs its output."""
    print(f"\n{'='*25}\nRUNNING: {' '.join(command)}\n{'='*25}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        with open(log_file, 'a') as f:
            for line in process.stdout:
                sys.stdout.write(line)
                f.write(line)
        process.wait()
        if process.returncode != 0:
            error_msg = f"!!! Step failed with return code {process.returncode} !!!"
            print(error_msg)
            with open(log_file, 'a') as f:
                f.write(f"\n{error_msg}\n")
            return False
    except Exception as e:
        error_msg = f"!!! An exception occurred while running the step: {e} !!!"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write(f"\n{error_msg}\n")
        return False
    return True


def prepare_trade_data(systemDetails, now):
    """Loads daily trades, maps tokens to symbols, and filters for today."""
    try:
        signals_df = pd.read_csv('reports/trades/daily_trades.csv')
    except FileNotFoundError:
        print("ERROR: daily_trades.csv not found. Skipping trade processing.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    token_list = signals_df['instrument_token'].unique()
    token_to_symbol = {}
    for token in token_list:
        result = systemDetails.run_query_limit(f"Select distinct tradingsymbol from kiteconnect.instruments_zerodha where instrument_token = {token}")
        if result:
            token_to_symbol[token] = result[0]
            
    signals_df['tradingsymbol'] = signals_df['instrument_token'].map(token_to_symbol)
    
    # Use hardcoded dates for testing if needed, otherwise use current date
    # exit_df = signals_df[signals_df['exit_date'] == '2025-07-18']
    # entry_df = signals_df[signals_df['entry_date'] == '2025-07-18']
    exit_df = signals_df[signals_df['exit_reason'] == 'exit_today']
    entry_df = signals_df[signals_df['exit_reason'] == 'enter_today']
    active_df = signals_df[(signals_df['entry_date'] <= now.strftime('%Y-%m-%d')) & (signals_df['exit_reason'] == 'Active')]
    
    print(f"INFO: Found {len(entry_df)} new entry signals and {len(exit_df)} new exit signals for today.")
    return signals_df, entry_df, exit_df, active_df

def ask_gemini(entry_df, order_placement):
    """Processes entry signals, gets Gemini opinion, and places buy orders."""
    if entry_df.empty:
        order_placement.send_telegram_message("No New Trades Today")

    print(f"--- Processing {len(entry_df)} Entries ---")
    order_placement.send_telegram_message("New Trades:")
    all_results = []
    
    for _, row in entry_df.iterrows():
        tradingsymbol = row['tradingsymbol']
        gemini_result = run_gemini_bridge(tradingsymbol)
        all_results.extend(gemini_result)

    for item in all_results:
        order_placement.send_telegram_message(json.dumps(item, indent=4))


def main():
    """Main pipeline execution function."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(LOG_DIR, f"live_run_{timestamp}.log")
    
    python_executable = '/opt/anaconda3/envs/KiteConnect/bin/python'
    print(f"--- Starting Full Pipeline Run ---")
    print(f"Logging output to: {log_file}")
    
    config = load_config()
    order_placement = OrderPlacement()
    systemDetails = system_initialization()
    callKite = kiteAPIs()

    
    order_placement.send_telegram_message(f"Starting Trade Generation after token refresh for date {config['data']['test_end_date']}")
    try:
        systemDetails.init_trading()
        # order_placement.send_telegram_message(f"Starting Live Run for date {config['data']['test_end_date']}")
    except Exception as e:
        print(f"Initial connection failed: {e}")
        # Attempt a hard refresh on initial failure
        try:
            systemDetails.hard_refresh_access_token()
            order_placement.init_trading()
            order_placement.send_telegram_message(f"Starting Trade Generation after token refresh for date {config['data']['test_end_date']}")
        except Exception as refresh_e:
            print(f"FATAL: Hard refresh on startup failed: {refresh_e}")
            order_placement.send_telegram_message(f"FATAL: Hard refresh on startup failed: {refresh_e}")
            sys.exit(1)

 
    now = datetime.now()
    print(f"\n{'='*30}\n--- Starting New Trading Cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{'='*30}")
    

    try:
        # --- 1. Run Data Pipeline Steps ---
        # These steps are assumed to be necessary on each loop iteration
        run_step([python_executable,  '-u', 'src/auto_update_date.py'], log_file)
        run_step([python_executable,  '-u', 'src/data_backfill/data_backfill_daily.py'], log_file)
        run_step([python_executable,  '-u', 'src/data_pipeline.py'], log_file)
        run_step([python_executable,  '-u', 'src/feature_generator.py'], log_file)
        run_step([python_executable,  '-u', 'src/pattern_feature_generator.py'], log_file)
        run_step([python_executable,  '-u', 'src/merge_features.py'], log_file)
        run_step([python_executable,  '-u', 'src/trade_generator.py'], log_file)

        signals_df, entry_df, exit_df, active_df = prepare_trade_data(systemDetails, now)
        ask_gemini(entry_df, order_placement)
        
    except KeyboardInterrupt:
        print("\nTrade Generation stopped by user.")
        order_placement.send_telegram_message("Trade Generation stopped manually.")
    except Exception as e:
        if 'Incorrect `api_key` or `access_token`' in str(e) or 'Invalid `api_key`' in str(e):
            print(f"ERROR: Access token expired: {e}")
            print("INFO: Attempting to refresh token and re-initialize.")
            try:
                systemDetails.hard_refresh_access_token()
                order_placement = OrderPlacement()
                order_placement.init_trading()
                callKite = kiteAPIs()
                print("SUCCESS: Successfully refreshed token and re-initialized services. Continuing monitoring.")
            except Exception as refresh_e:
                print(f"FATAL: Failed to refresh access token: {refresh_e}")
                order_placement.send_telegram_message(f"FATAL: Failed to refresh token: {refresh_e}")
        else:
            print(f"ERROR: An unexpected error occurred in the main loop: {e}")
            order_placement.send_telegram_message(f"ERROR in Trade Generation: {e}")
    
    
    print(f"\n--- Trade Generation Finished Successfully ---")
    print(f"Full log available at: {log_file}")

if __name__ == "__main__":
    main()