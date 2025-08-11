# /opt/anaconda3/bin/activate && conda activate /opt/anaconda3/envs/KiteConnect

# 0 16 * * * cd /Users/shashankbaveja/Downloads/Personal/KiteConnectAPI/trading_setup && /opt/anaconda3/envs/KiteConnect/bin/python data_backfill.py >> /Users/shashankbaveja/Downloads/Personal/KiteConnectAPI/trading_setup/data_backfill_cron.log 2>&1

# Add project root to sys.path to resolve module imports
import sys
import os
from IPython import embed;
from kiteconnect import KiteConnect, KiteTicker
import mysql
import mysql.connector as sqlConnector
import datetime
from selenium import webdriver
import os
from pyotp import TOTP
import ast
import time
import pandas as pd
from sqlalchemy import create_engine
import pymysql
from myKiteLib import system_initialization, kiteAPIs, OrderPlacement
import logging
import json
from datetime import date, timedelta, datetime, time
from kiteconnect.exceptions import KiteException  # Import KiteConnect exceptions
import requests # Import requests for ReadTimeout
import numpy as np

from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


def get_trade_dates(sys_init):
    query = "Select tradingsymbol, min(fill_timestamp) as trade_date_db from kiteconnect.trades where transaction_type = 'BUY' and fill_timestamp >= curdate() - interval 21 day group by 1;"
    df = sys_init.run_query_full(query)
    return df

def get_holdings(callKite, sys_init, order_placement):
    print("INFO: Fetching holdings...")
    holdings_data = callKite.extract_holdings_data()
    positions_data = callKite.extract_positions_data() 
    all_data = holdings_data + positions_data
    # print(all_data)
    df = pd.DataFrame(all_data)
    dates = get_trade_dates(sys_init)

    exclude_symbols = ['IDEA', 'IDFCFIRSTB', 'YESBANK']
    if not df.empty:
        df = df[~df['tradingsymbol'].isin(exclude_symbols)]
    
    merged_df = pd.merge(df, dates, on='tradingsymbol', how='left')
    merged_df['trade_date_db'] = pd.to_datetime(merged_df['trade_date_db'])

    condition = merged_df['data_source'] == 'positions'
    holding_days = (pd.Timestamp.now() - merged_df['trade_date_db']).dt.days
    merged_df['holding_period'] = np.where(
        condition,       # If data_source is 'positions'...
        0,               # ...then set holding_period to 0.
        holding_days     # Otherwise, set it to the calculated holding days.
    )

    token_list = merged_df['instrument_token'].unique().tolist()
    ltp_dict = order_placement.get_ltp_live(token_list)
    # print(token_list)
    # print(ltp_dict)
    merged_df['ltp'] = merged_df['instrument_token'].map(ltp_dict)
    merged_df['pnl'] = merged_df['quantity']*(merged_df['ltp'] - merged_df['average_price'])
    merged_df['pnl_percent'] = (100*merged_df['ltp']/merged_df['average_price'])-100
    merged_df = merged_df.drop(columns=['trade_date', 'last_price'])
    return merged_df


if __name__ == "__main__":

    BACKFILL_INTERVAL = 'day'
    BACKFILL_DAYS = 1
    
    today_date = date.today()
    
    end_dt_backfill = datetime.combine(today_date, time(23, 59, 59))
    
    start_date_val = today_date - timedelta(days=BACKFILL_DAYS)
    start_dt_backfill = datetime.combine(start_date_val, time(0, 0, 0))

    print(f"starting system_init")
    systemDetails = system_initialization()
    systemDetails.init_trading()
    callKite = kiteAPIs()
    order_placement = OrderPlacement()
    trades = callKite.get_trades()
    print("Trades Updated in DB!")
    
    
    active_trades = get_holdings(callKite, systemDetails, order_placement)
    print(active_trades)

    # breakpoint()
    # systemDetails.run_query_limit(f"Call trades_PnL();") 
    # PnL = systemDetails.run_query_full(f"Select * from kiteconnect.trades_PnL;")
    
    # Pnl, metrics = systemDetails.GetPnL()
    # Pnl.to_csv('pnl.csv')
    # print(metrics)
    # order_placement.send_telegram_message(f"Metrics: {metrics}")
    
    tokenList = active_trades['instrument_token'].tolist()
    
    # tokenList_old = [256265] ## NIFTY INDEX
    # tokenList_old.extend(callKite.get_instrument_all_tokens('EQ'))

    # breakpoint()

    try:
        print(f"Deleting data for {start_dt_backfill} to {end_dt_backfill}")
        result = systemDetails.DeleteData("Delete from kiteconnect.historical_data_day where timestamp >= CURDATE()")
        print(f"Result: {result}")
        df = callKite.getHistoricalData(start_dt_backfill,  end_dt_backfill, tokenList, BACKFILL_INTERVAL)
    except (KiteException, requests.exceptions.ReadTimeout) as e:
        print(f"Error fetching historical data: {e}")
        logging.error(f"Error fetching historical data: {e}")
        df = pd.DataFrame() # Initialize an empty DataFrame or handle as needed
    