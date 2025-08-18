import os
import sys
import subprocess
from datetime import datetime
import time
import pandas as pd
import json
from kiteconnect.exceptions import KiteException, InputException
import google.generativeai as genai
import requests

# --- Configuration & Imports ---
LOG_DIR = 'logs/live_monitoring'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myKiteLib import OrderPlacement, kiteAPIs, system_initialization
from src.utils.live_trading_helpers import (
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


def prepare_trade_data(systemDetails, now):
    """Loads daily trades, maps tokens to symbols, and filters for today."""
    try:
        # Prefer v3 output; fallback to legacy filename if needed
        signals_df = pd.read_csv('reports/trade_log.csv')
    except FileNotFoundError:
        print("ERROR: trade_log.csv not found. Skipping trade processing.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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


def retry_with_backoff(func, max_retries=3, backoff_factor=2, *args, **kwargs):
    """
    Retry a function with exponential backoff on connection errors.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        Function result if successful, None if all retries failed
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt == max_retries:
                print(f"FATAL: All {max_retries + 1} connection attempts failed. Last error: {e}")
                raise e
            
            wait_time = backoff_factor ** attempt
            print(f"Connection error on attempt {attempt + 1}/{max_retries + 1}: {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            # For non-connection errors, don't retry
            print(f"Non-connection error occurred: {e}")
            raise e


def reconnect_with_retry(systemDetails, order_placement, max_retries=3):
    """
    Attempt to reconnect to the API with retry logic.
    
    Returns:
        True if reconnection successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            print(f"Attempting to refresh access token (attempt {attempt + 1}/{max_retries})...")
            systemDetails.hard_refresh_access_token()
            order_placement.init_trading()
            print("Successfully reconnected to API")
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            print(f"Reconnection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {wait_time} seconds before next attempt...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"Unexpected error during reconnection: {e}")
            return False
    
    print(f"Failed to reconnect after {max_retries} attempts")
    return False


def main():
    """Main pipeline execution function."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(LOG_DIR, f"live_run_{timestamp}.log")
    
    python_executable = '/opt/anaconda3/envs/KiteConnect/bin/python'
    print(f"--- Starting Live Monitoring ---")
    print(f"Logging output to: {log_file}")
    
    config = load_config()
    order_placement = OrderPlacement()
    systemDetails = system_initialization()
    callKite = kiteAPIs()

    
    order_placement.send_telegram_message(f"Starting Live Monitoring after token refresh for date {config['data']['test_end_date']}")
    try:
        systemDetails.init_trading()
        # order_placement.send_telegram_message(f"Starting Live Run for date {config['data']['test_end_date']}")
    except Exception as e:
        print(f"Initial connection failed: {e}")
        # Attempt a hard refresh on initial failure
        try:
            systemDetails.hard_refresh_access_token()
            order_placement.init_trading()
            order_placement.send_telegram_message(f"Starting Live Monitoring after token refresh for date {config['data']['test_end_date']}")
        except Exception as refresh_e:
            print(f"FATAL: Hard refresh on startup failed: {refresh_e}")
            order_placement.send_telegram_message(f"FATAL: Hard refresh on startup failed: {refresh_e}")
            sys.exit(1)

 
    now = datetime.now() 
    print(f"\n{'='*30}\n--- Starting New Trading Cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{'='*30}")

    market_start_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_trade_end_time = now.replace(hour=13, minute=25, second=0, microsecond=0)
    market_end_time = now.replace(hour=15, minute=31, second=0, microsecond=0)
# --- 2. Fetch Live and Signal Data ---
    
    # Use retry logic for initial data fetching
    try:
        active_trades = retry_with_backoff(get_holdings, 3, 2, callKite, systemDetails, order_placement)
        print(f"Active Trades: {active_trades}")
        signals_df, entry_df, exit_df, active_df = prepare_trade_data(systemDetails, datetime.now())
    except Exception as e:
        print(f"FATAL: Could not fetch initial data after retries: {e}")
        order_placement.send_telegram_message(f"FATAL: Could not fetch initial data: {e}")
        sys.exit(1)

    # check_holding_mismatch(exit_df, active_df, active_trades, order_placement)
    trade_params_df = pd.concat([entry_df, active_df]).drop_duplicates(subset=['tradingsymbol'])

    print(f"Entry DF: {entry_df}")
    print(f"Exit DF: {exit_df}")
    print(f"Active DF: {active_df}")
   
    
    while True:
        now = datetime.now()
        if market_start_time <= now <= market_end_time:
            try:
                print("\n--- Running Live TP/SL Check ---")
                
                # Use retry logic for fetching holdings
                live_holdings_df = retry_with_backoff(get_holdings, 3, 2, callKite, systemDetails, order_placement)

                if live_holdings_df.empty:
                    print("No live holdings to monitor.")
                elif not trade_params_df.empty:
                    # Find which of our live holdings we need to monitor
                    monitored_trades = pd.merge(live_holdings_df, trade_params_df, on='tradingsymbol', how='inner')
                    
                    if not monitored_trades.empty:
                        # Fetch LTP for all monitored instruments at once with retry
                        instruments = [row['instrument_token_x'] for _, row in monitored_trades.iterrows()]
                        ltp_data = retry_with_backoff(order_placement.get_ltp_live, 3, 2, instruments)

                        for _, trade in monitored_trades.iterrows():
                            instrument_id = trade['instrument_token_x']
                            ltp = ltp_data[instrument_id]
                            if ltp is None:
                                print(f"Could not get LTP for {trade['tradingsymbol']}.")
                                continue
                            
                            tp_price = trade['average_price'] * (1 + trade['tp_thresh'])
                            sl_price = trade['average_price'] * (1 - trade['sl_thresh'])
                            
                            exit_reason = None
                            
                            if ltp >= tp_price:
                                exit_reason = 'TP_HIT'
                            elif ltp <= sl_price:
                                exit_reason = 'SL_HIT'
                            
                            if exit_reason:
                                quantity_to_exit = int(trade['quantity'])
                                print(f"ALERT: {exit_reason} for {trade['tradingsymbol']} at price {ltp}. Exiting {quantity_to_exit} shares.")
                                
                                # Use retry logic for placing orders
                                try:
                                    retry_with_backoff(
                                        order_placement.place_market_order_live, 3, 2,
                                        trade['tradingsymbol'], trade['exchange'], 'SELL', quantity_to_exit, 'CNC', exit_reason
                                    )
                                    order_placement.send_telegram_message(
                                        f"Exiting trade for {trade['tradingsymbol']} ({quantity_to_exit} qty) due to: {exit_reason} at {ltp}"
                                    )
                                    # Remove from params so we don't try to exit it again in this session
                                    trade_params_df = trade_params_df[trade_params_df['tradingsymbol'] != trade['tradingsymbol']]
                                    print(f"Removed {trade['tradingsymbol']} from monitoring list for this session.")
                                    break # Re-run the holdings check after exit
                                except Exception as order_e:
                                    print(f"Failed to place exit order for {trade['tradingsymbol']} after retries: {order_e}")
                                    order_placement.send_telegram_message(f"FAILED to exit {trade['tradingsymbol']}: {order_e}")
                                    
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_e:
                print(f"Connection error during monitoring loop: {conn_e}")
                order_placement.send_telegram_message(f"Connection error in monitoring loop: {conn_e}")
                
                # Attempt to reconnect
                if reconnect_with_retry(systemDetails, order_placement):
                    print("Reconnection successful, continuing monitoring...")
                    order_placement.send_telegram_message("Reconnected successfully, monitoring resumed")
                else:
                    print("FATAL: Could not reconnect. Shutting down monitoring.")
                    order_placement.send_telegram_message("FATAL: Could not reconnect. Monitoring stopped.")
                    break
                    
            except Exception as e:
                print(f"Unexpected error during live monitoring loop: {e}")
                order_placement.send_telegram_message(f"Unexpected error in monitoring loop: {e}")
                
                # For unexpected errors, try a simple reconnection
                try:
                    systemDetails.hard_refresh_access_token()
                    order_placement.init_trading()
                    print("Token refreshed successfully")
                except Exception as refresh_e:
                    print(f"Failed to refresh token: {refresh_e}")
                    order_placement.send_telegram_message(f"Failed to refresh token: {refresh_e}")
                    break
            
            print(f"Check complete. Sleeping for 5 minutes...")
            time.sleep(60)
        else:
            print("Market is closed. Stopping live monitoring.")
            break
            

    print(f"\n--- Live Monitoring Finished Successfully ---")
    print(f"Full log available at: {log_file}")

    
if __name__ == '__main__':
    main() 