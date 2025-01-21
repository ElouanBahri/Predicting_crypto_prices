from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf


def create_features_from_past(
    df, column=["close", "open", "high", "low", "volume"], window_size=4
):
    # Drop multiple columns
    df = df.drop(
        columns=[
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
    )

    df = df.copy()
    for i in range(1, window_size + 1):
        df[f"lag_close{i}"] = df[column[0]].shift(i)
        df[f"lag_open{i}"] = df[column[1]].shift(i)
        df[f"lag_high{i}"] = df[column[2]].shift(i)
        df[f"lag_low{i}"] = df[column[3]].shift(i)
        df[f"lag_volume{i}"] = df[column[4]].shift(i)

    # Drop rows with NaN due to lagging
    df = df.dropna()

    return df


def filter_data_by_year_month(df, years):
    # Ensure 'timestamp' column is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Apply filtering based on years and months
    filtered_df = df[df["timestamp"].dt.year.isin(years)]

    return filtered_df


def create_X_y(df):
    y = df["close"]

    # Create X by dropping specified columns
    X = df.drop(columns=["low", "high", "volume", "close", "timestamp", "open"])

    return X, y
