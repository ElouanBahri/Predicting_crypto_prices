import os
import sys
from datetime import datetime

import pandas as pd
import requests

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from src.config import binanceConfig

url_historical_prices = binanceConfig.URL_HISTORICAL_PRICES
columns_historical_prices = binanceConfig.COLUMNS_HISTORICAL_PRICES
limit = binanceConfig.MAX_LIMIT


def convert_date_to_timestamp(date_str: str):
    dt_object = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt_object.timestamp() * 1000)


def fetch_data(symbol: str, interval: str, start_time: float):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": limit,
    }
    print("Retrieving data from " + str(datetime.fromtimestamp(start_time / 1000.0)))
    response = requests.get(url_historical_prices, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error with request : {response.status_code}")
        return None


def get_all_data(symbol: str, interval: str, start_date: str):
    start_time = convert_date_to_timestamp(start_date)
    all_data = []
    while True:
        data = fetch_data(symbol, interval, start_time)
        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][0] + 1
    return all_data


def process_data(data: list):
    df = pd.DataFrame(data, columns=columns_historical_prices)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df.set_index("timestamp", inplace=False)
    return df


def get_binance_data(symbol: str, interval: str, start_date: str):
    data = get_all_data(symbol, interval, start_date)
    if data:
        df = process_data(data)
        return df
    else:
        print("No data")
        return None
