from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf


def prep_for_RNN(df, target_column, timesteps=3):
    """
    Transforms a DataFrame into sequences of input-output pairs for RNN.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing features and target column.
        target_column (str): Name of the target column to predict.
        timesteps (int): Number of timesteps to use as input.

    Returns:
        X (np.array): 3D array of input sequences (samples, timesteps, features).
        y (np.array): 1D array of target values corresponding to each sequence.
    """
    X, y = [], []

    df = df.drop(
        columns=[
            "close_time",
            "timestamp",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
    )
    for i in range(len(df) - timesteps):
        input_sequence = df.iloc[
            i : i + timesteps
        ].values  # Use all columns, including 'close'
        target_value = df.iloc[i + timesteps][
            target_column
        ]  # Target value is the next 'close'

        X.append(input_sequence)
        y.append(target_value)

    return np.array(X), np.array(y)


def create_features_for_next_prediction(
    df, columns=["close", "open", "high", "low", "volume"], window_size=4
):
    # Copy only the last `window_size` rows
    last_rows = df.iloc[-window_size:].copy()

    # Create a new DataFrame to store the new row with lagged features
    new_row = {}

    for i in range(1, window_size + 1):
        for col in columns:
            # Generate lagged features
            new_row[f"lag_{col}{i}"] = last_rows.iloc[-i][col]

    # Convert the new row into a DataFrame
    new_row_df = pd.DataFrame([new_row])

    return new_row_df


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
