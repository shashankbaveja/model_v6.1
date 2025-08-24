import os
import sys
from datetime import datetime
import time
import pandas as pd
from kiteconnect.exceptions import KiteException, InputException
import math
 
# --- Configuration & Imports ---
LOG_DIR = 'logs/live_rebalancing'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myKiteLib import OrderPlacement, kiteAPIs, system_initialization
from src.utils.live_trading_helpers import (
    load_config, get_holdings
)

liquid_token = '139286788'
liquid_symbol = 'LIQUIDCASE'

def prepare_trade_data(systemDetails, now):
    """Loads daily trades, maps tokens to symbols, and filters for today."""
    try:
        # Prefer v3 output; fallback to legacy filename if needed
        signals_df = pd.read_csv('reports/trade_log.csv')
    except FileNotFoundError:
        print("ERROR: trade_log.csv not found. Skipping trade processing.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    token_list = signals_df['instrument_token'].unique()
    token_to_symbol = {}
    for token in token_list:
        result = systemDetails.run_query_limit(f"Select distinct tradingsymbol from kiteconnect.instruments_zerodha where instrument_token = {token}")
        if result:
            token_to_symbol[token] = result[0]
            
    signals_df['tradingsymbol'] = signals_df['instrument_token'].map(token_to_symbol)
    exit_df = signals_df[signals_df['exit_reason'] == 'exit_today']
    entry_df = signals_df[signals_df['exit_reason'] == 'enter_today']
    signals_df['entry_amount'] = signals_df['num_shares'] * signals_df['entry_price']
    gross_amount_needed = signals_df[signals_df['exit_reason'] == 'enter_today']['entry_amount'].sum()
    active_df = signals_df[(signals_df['entry_date'] <= now.strftime('%Y-%m-%d')) & (signals_df['exit_reason'] == 'Active')]
    
    print(f"INFO: Found {len(entry_df)} new entry signals and {len(exit_df)} new exit signals for today.")
    return signals_df, entry_df, exit_df, active_df, gross_amount_needed


def _summarize_kite_error_message(err_text: str) -> str:
    text = str(err_text) if err_text is not None else ''
    lower_text = text.lower()
    if 'market orders are blocked' in lower_text or 'market orders are blocked' in text:
        return 'MARKET orders are blocked'
    if 'insufficient funds' in lower_text:
        return 'Insufficient funds'
    if 'markets are closed right now' in lower_text:
        return 'Markets are closed right now'
    # Fallback to exact exception text
    return text

def _notify_order_failure(order_placement, symbol: str, side: str, err: Exception):
    reason = _summarize_kite_error_message(err)
    msg = f"{symbol} {side} order failed: {reason}"
    print(msg)
    order_placement.send_telegram_message(msg)

def rebalance_portfolio(order_placement, systemDetails, live_holdings_df, entry_df, exit_df, active_df, gross_amount_needed, active_trades):
    """
    Reconciles the live portfolio state with the target state from signal files.
    - Enters trades that are in entry/active lists but not in live holdings.
    - Exits trades that are in live holdings and also in the exit list.
    - Exits trades that are in live holdings but not in any internal list (mismatch).
    """
    print("\n--- Reconciling Portfolio State ---")

    # Ensure dataframes have 'tradingsymbol' column
    for df_name, df in [('live_holdings_df', live_holdings_df), ('entry_df', entry_df), ('exit_df', exit_df), ('active_df', active_df)]:
        if 'tradingsymbol' not in df.columns and not df.empty:
            print(f"WARN: '{df_name}' is missing 'tradingsymbol' column. Skipping.")
            # Or handle error appropriately
            return

    # Create sets of trading symbols for efficient comparison
    held_symbols = set(live_holdings_df['tradingsymbol']) if not live_holdings_df.empty else set()
    held_symbols = held_symbols - {liquid_symbol}
    entry_symbols = set(entry_df['tradingsymbol']) if not entry_df.empty else set()
    active_symbols = set(active_df['tradingsymbol']) if not active_df.empty else set()
    exit_symbols = set(exit_df['tradingsymbol']) if not exit_df.empty else set()

    target_symbols = entry_symbols.union(active_symbols)
    all_internal_symbols = target_symbols.union(exit_symbols)


    # 2. Determine trades to EXIT
    planned_exits = held_symbols.intersection(exit_symbols)
    mismatch_exits = held_symbols - all_internal_symbols
    symbols_to_exit = planned_exits.union(mismatch_exits)

    if symbols_to_exit:
        print(f"--- Processing {len(symbols_to_exit)} Exits ---")
        for symbol in symbols_to_exit:
            reason = 'PLANNED_EXIT' if symbol in planned_exits else 'MISMATCH_CLEANUP'
            
            # Get quantity and exchange from live holdings
            holding_details = live_holdings_df[live_holdings_df['tradingsymbol'] == symbol].iloc[0]
            quantity = int(holding_details['quantity'])
            exchange = holding_details['exchange']

            print(f"Exiting trade for {symbol} ({quantity} qty) due to: {reason}")
            try:
                order_placement.place_market_order_live(symbol, exchange, 'SELL', quantity, 'CNC', reason, notify_on_error=False)
                order_placement.send_telegram_message(f"Exiting trade for {symbol} ({quantity} qty) due to: {reason}")
            except Exception as e:
                _notify_order_failure(order_placement, symbol, 'SELL', e)
    
    
    
    get_back_liquidity(active_trades, liquid_token, order_placement, systemDetails, gross_amount_needed)
    # 1. Determine trades to ENTER
    symbols_to_enter = target_symbols - held_symbols
    if symbols_to_enter:
        print(f"--- Processing {len(symbols_to_enter)} Entries ---")
        # Combine entry and active dfs to get trade parameters
        enter_params_df = pd.concat([entry_df, active_df]).drop_duplicates(subset=['tradingsymbol'])
        
        for symbol in symbols_to_enter:
            trade_details = enter_params_df[enter_params_df['tradingsymbol'] == symbol].iloc[0]
            quantity = int(trade_details['num_shares']) # Assuming 'num_shares' is the column for quantity
            print(f"Entering trade for {symbol}, Quantity: {quantity}")
            
            try:
                # This logic is copied from process_entries to handle NSE/BSE
                order_id = order_placement.place_market_order_live(symbol, 'NSE', 'BUY', quantity, 'CNC', 'Recon_Entry', notify_on_error=False)
                message = f"Order placed successfully for {symbol} on NSE. Order ID: {order_id}"
                print(message)
                order_placement.send_telegram_message(message)
            except InputException as e:
                if 'The instrument you are placing an order for has either expired or does not exist' in str(e):
                    print(f"Instrument {symbol} not found on NSE. Attempting on BSE.")
                    try:
                        order_id_bse = order_placement.place_market_order_live(symbol, 'BSE', 'BUY', quantity, 'CNC', 'Recon_Entry', notify_on_error=False)
                        message = f"Order placed successfully for {symbol} on BSE. Order ID: {order_id_bse}"
                        print(message)
                        order_placement.send_telegram_message(message)
                    except KiteException as bse_e:
                        _notify_order_failure(order_placement, symbol, 'BUY', bse_e)
            except KiteException as e:
                _notify_order_failure(order_placement, symbol, 'BUY', e)

   
    if not symbols_to_enter and not symbols_to_exit:
        print("OK: Portfolio is already in sync with signals. No trades needed.")

def report_portfolio_mismatches(order_placement, live_holdings_df, entry_df, exit_df, active_df):
    """
    Compute and report mismatches without placing trades.
    Sends a Telegram message for each mismatch found.
    """
    print("\n--- Reporting Portfolio Mismatches (no trades) ---")

    held_symbols = set(live_holdings_df['tradingsymbol']) if not live_holdings_df.empty else set()
    held_symbols = held_symbols - {liquid_symbol}
    entry_symbols = set(entry_df['tradingsymbol']) if not entry_df.empty else set()
    active_symbols = set(active_df['tradingsymbol']) if not active_df.empty else set()
    exit_symbols = set(exit_df['tradingsymbol']) if not exit_df.empty else set()

    target_symbols = entry_symbols.union(active_symbols)
    all_internal_symbols = target_symbols.union(exit_symbols)

    should_not_exist = held_symbols - all_internal_symbols
    should_exist = target_symbols - held_symbols

    if not should_not_exist and not should_exist:
        print("No mismatches detected after reconciliation.")
        return

    for symbol in sorted(should_not_exist):
        msg = f"tradingsymbol {symbol} should not exist but it does"
        print(msg)
        order_placement.send_telegram_message(msg)

    for symbol in sorted(should_exist):
        msg = f"tradingsymbol {symbol} should exist but it does not. No more trades during this check."
        print(msg)
        order_placement.send_telegram_message(msg)

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
        except Exception as e:
            if attempt == max_retries:
                print(f"FATAL: All {max_retries + 1} connection attempts failed. Last error: {e}")
                raise e
            
            wait_time = backoff_factor ** attempt
            print(f"Connection error on attempt {attempt + 1}/{max_retries + 1}: {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

def get_back_liquidity(active_trades, liquid_token, order_placement, systemDetails, gross_amount_needed):
    margins_json = systemDetails.kite.margins()
    equity = margins_json.get('equity', {}) or {}
    available = equity.get('available', {}) or {}
    # cash_available = available.get('cash')
    cash_available = available.get('live_balance')
    liquid_token_int = int(liquid_token)
    liquidcash_ltp = order_placement.get_ltp_live([liquid_token]).get(liquid_token_int)

    liquidcash_quantity = 0
    if active_trades is not None and not active_trades.empty:
        # liquid_mask = active_trades['instrument_token'] == liquid_token_int
        liquidcash_quantity = active_trades[active_trades['tradingsymbol'] == liquid_symbol]['quantity'].sum()
    liquidcash_value = liquidcash_quantity * liquidcash_ltp if liquidcash_quantity > 0 else 0
    if liquidcash_value < gross_amount_needed:
        message = f"FATAL: Not enough liquidity to cover the required amount. Required: {gross_amount_needed}, Available: {liquidcash_value}"
        print(message)
        order_placement.send_telegram_message(message)
        sys.exit(1)

    if cash_available < gross_amount_needed - 5000:
        required_amount = gross_amount_needed - cash_available + 5000
        required_liquid_quantity = math.ceil(required_amount/liquidcash_ltp)
        try:
            # This logic is copied from process_entries to handle NSE/BSE
            order_id = order_placement.place_market_order_live(liquid_symbol, 'NSE', 'SELL', required_liquid_quantity, 'CNC', 'Recon_Entry', notify_on_error=False)
            message = f"Order placed successfully for {liquid_symbol} on NSE. Order ID: {order_id}"
            print(message)
            order_placement.send_telegram_message(message)
        except Exception as e:
            _notify_order_failure(order_placement, liquid_symbol, 'SELL', e)


def put_back_liquidity(liquid_symbol, liquid_token, systemDetails, order_placement):
    margins_json = systemDetails.kite.margins()
    equity = margins_json.get('equity', {}) or {}
    available = equity.get('available', {}) or {}
    # cash_available = available.get('cash')
    cash_available = available.get('live_balance')
    liquidcash_ltp = order_placement.get_ltp_live([liquid_token]).get(int(liquid_token))
    liquidcash_quantity = math.floor(cash_available/liquidcash_ltp)
    
    try:
        # This logic is copied from process_entries to handle NSE/BSE
        order_id = order_placement.place_market_order_live(liquid_symbol, 'NSE', 'BUY', liquidcash_quantity, 'CNC', 'Recon_Entry', notify_on_error=False)
        message = f"Order placed successfully for {liquid_symbol} on NSE. Order ID: {order_id}"
        print(message)
        order_placement.send_telegram_message(message)
    except Exception as e:
        _notify_order_failure(order_placement, liquid_symbol, 'BUY', e)

def main():
    """Main pipeline execution function."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(LOG_DIR, f"live_run_{timestamp}.log")
    
    
    print(f"--- Starting Rebalancing ---")
    print(f"Logging output to: {log_file}")
    
    config = load_config()
    order_placement = OrderPlacement()
    systemDetails = system_initialization()
    callKite = kiteAPIs()

    
    order_placement.send_telegram_message(f"Starting Rebalancing")
    try:
        systemDetails.init_trading()
    except Exception as e:
        print(f"Initial connection failed: {e}")
        # Attempt a hard refresh on initial failure
        try:
            systemDetails.hard_refresh_access_token()
            systemDetails.init_trading()
            order_placement.send_telegram_message(f"Starting Live rebalancing after token refresh for date {config['data']['test_end_date']}")
        except Exception as refresh_e:
            print(f"FATAL: Hard refresh on startup failed: {refresh_e}")
            order_placement.send_telegram_message(f"FATAL: Hard refresh on startup failed: {refresh_e}")
            sys.exit(1)

# --- 2. Fetch Live and Signal Data ---
    
    # Use retry logic for initial data fetching
    try:
        active_trades = retry_with_backoff(get_holdings, 3, 2, callKite, systemDetails, order_placement)
        print(f"Active Trades: {active_trades}")
        signals_df, entry_df, exit_df, active_df, gross_amount_needed = prepare_trade_data(systemDetails, datetime.now())


    except Exception as e:
        print(f"FATAL: Could not fetch initial data after retries: {e}")
        order_placement.send_telegram_message(f"FATAL: Could not fetch initial data: {e}")
        sys.exit(1)

    # check_holding_mismatch(exit_df, active_df, active_trades, order_placement)

            
    try:
        # Initial reconciliation before rebalancing loop
        rebalance_portfolio(order_placement, systemDetails, active_trades, entry_df, exit_df, active_df, gross_amount_needed, active_trades)

    except Exception as e:
        print(f"ERROR: An unexpected error occurred in the main loop: {e}")
        order_placement.send_telegram_message(f"ERROR in Live rebalancing: {e}")

    try:
        # Re-fetch holdings and report mismatches (no trades) before loop
        refreshed_holdings = retry_with_backoff(get_holdings, 3, 2, callKite, systemDetails, order_placement)
        report_portfolio_mismatches(order_placement, refreshed_holdings, entry_df, exit_df, active_df)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in the main loop: {e}")
        order_placement.send_telegram_message(f"ERROR in Live rebalancing: {e}")
    
    
    put_back_liquidity(liquid_symbol, liquid_token, systemDetails, order_placement)

    print(f"\n--- Live rebalancing Finished Successfully ---")
    print(f"Full log available at: {log_file}")

    
if __name__ == '__main__':
    main() 