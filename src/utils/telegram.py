import json
import datetime
import os
import time
import requests



def send_telegram_message(message: str):
        """
        Sends a message to the configured Telegram chat.
        Args:
            message (str): The message text to send.
        """
        telegram_token = '8135376207:AAFoMWbyucyPPEzc7CYeAMTsNZfqHWYDMfw' # Renamed to avoid conflict
        telegram_chat_id = "-4653665640"

        if not telegram_token or not telegram_chat_id:
            print("OrderPlacement: Telegram token or chat ID not configured. Cannot send message.")
            return
        
        telegram_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        params = {
            'chat_id': telegram_chat_id,
            'text': message
        }
        try:
            response = requests.get(telegram_url, params=params, timeout=10) # Added timeout
            response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
            print(f"OrderPlacement: Telegram message sent successfully: '{message[:50]}...'")
        except requests.exceptions.RequestException as e:
            print(f"OrderPlacement: Error sending Telegram message: {e}")
        except Exception as e:
            print(f"OrderPlacement: A general error occurred sending Telegram message: {e}")