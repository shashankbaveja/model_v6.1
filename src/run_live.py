import os
import sys
import subprocess
from datetime import datetime
import time
import pandas as pd
import json
from kiteconnect.exceptions import KiteException, InputException

# --- Configuration & Imports ---
LOG_DIR = 'logs'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myKiteLib import OrderPlacement, kiteAPIs, system_initialization
from src.utils.live_trading_helpers import (
    load_config, get_holdings
)

# --- Environment Setup ---
from dotenv import load_dotenv
load_dotenv()

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
        run_step([python_executable,  '-u', 'src/signal_generator_v3.py'], log_file)
        run_step([python_executable,  '-u', 'src/trade_generator.py'], log_file)

        
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