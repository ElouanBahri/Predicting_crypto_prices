import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from datetime import datetime

import numpy as np
import pandas as pd

from src.config import data_config
from src.data import binance

general_filename = data_config.GENERAL_FILENAME
default_interval = data_config.DEFAULT_INTERVAL


def generate_filename(ticker: str):
    return general_filename + ticker


def save_data(list_symbol: list, interval: str, start_date: str):
    for ticker in list_symbol:
        print("Retrieve historical data for " + ticker)
        dt = binance.get_binance_data(ticker, interval, start_date)
        filename = generate_filename(ticker)
        dt.to_csv(filename)


def retrieve_data(symbol: str):
    filename = generate_filename(symbol)
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.close_time = pd.to_datetime(df.close_time)
    return df


def find_missing_data(df: pd.DataFrame, interval: str):
    first_date = df.index.min()
    last_date = df.index.max()
    if interval == "15m":
        date_range = pd.date_range(start=first_date, end=last_date, freq="15min")
    else:
        raise ValueError(
            f"Interval '{interval}' non supporté pour la vérification des dates."
        )
    missing_dates = set(date_range) - set(df.index)
    if len(missing_dates) == 0:
        print("No missing data")
    else:
        print(str(len(missing_dates)) + " missing data")
    return np.sort(list(missing_dates))


def aggregate_data(data: pd.DataFrame, timeframe: str):
    res = data.resample(timeframe).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "close_time": "last",
            "quote_asset_volume": "sum",
            "number_of_trades": "sum",
            "taker_buy_base_asset_volume": "sum",
            "taker_buy_quote_asset_volume": "sum",
            "ignore": "first",
        }
    )
    return res


def update_data(ticker: str, interval: str = default_interval):
    current_data = retrieve_data(ticker)
    last_date = current_data.index[-1]
    last_date_str = datetime.strftime(last_date, "%Y-%m-%d")
    new_data = binance.get_binance_data(ticker, interval, last_date_str)
    all_data = pd.concat([current_data, new_data], axis=0)
    all_data = all_data[~all_data.index.duplicated(keep="last")]
    all_data.sort_index(inplace=True)
    filename = generate_filename(ticker)
    all_data.to_csv(filename)
