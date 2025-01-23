from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


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
        # Extract the input sequence of the past 'timesteps' rows
        input_sequence = df.iloc[
            i : i + timesteps
        ].values  # Use all columns, including 'close'

        # Get the target value at the next timestep
        next_price = df.iloc[i + timesteps][target_column]

        # Compare the next price with the last price in the input sequence
        last_price_in_sequence = df.iloc[i + timesteps - 1][target_column]

        # Assign 1 if the price goes up, 0 if it goes down or stays the same
        target_value = 1 if next_price > last_price_in_sequence else 0

        # Append to the input (X) and target (y) lists
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


def potential_benef_strat_1(df, initial_ammount, leverage):
    """
    Calculate the potential benefit based on actual price changes with leverage,
    while deciding position (long/short) based on predicted prices.

    Parameters:
        df (pd.DataFrame): DataFrame containing `predicted_price` and `actual_price`.
        initial_ammount (float): The initial amount of money to invest.
        leverage (float): Leverage multiplier for trading.

    Returns:
        float: The final amount after all trades.
    """
    # Initialize the current amount with the initial amount
    current_amount = initial_ammount

    # Loop through each row except the last one
    for i in range(0, len(df) - 2):
        # Extract current and next row's predicted and actual prices

        current_actual_price = df.loc[i, "actual_price"]
        next_predicted_price = df.loc[i + 1, "predicted_price"]
        next_actual_price = df.loc[i + 1, "actual_price"]

        # Determine if we're going long or short
        if next_predicted_price > current_actual_price:
            # Long position
            price_change = (
                next_actual_price - current_actual_price
            ) / current_actual_price
        else:
            # Short position
            price_change = (
                current_actual_price - next_actual_price
            ) / current_actual_price

        # Calculate profit or loss with leverage
        profit_or_loss = current_amount * leverage * price_change

        print(profit_or_loss)

        # Update the current amount
        current_amount += profit_or_loss

    return current_amount


def trading_strat_2(df, initial_ammount, leverage, limit):
    """
    Calculate the potential benefit based on actual price changes with leverage,
    while deciding position (long/short) based on predicted prices.

    Parameters:
        df (pd.DataFrame): DataFrame containing `predicted_price` and `actual_price`.
        initial_ammount (float): The initial amount of money to invest.
        leverage (float): Leverage multiplier for trading.

    Returns:
        float: The final amount after all trades.
    """
    # Initialize the current amount with the initial amount
    current_amount = initial_ammount

    # Loop through each row except the last one
    for i in range(0, len(df) - 2):
        # Extract current and next row's predicted and actual prices

        current_actual_price = df.loc[i, "actual_price"]
        next_predicted_price = df.loc[i + 1, "predicted_price"]
        next_actual_price = df.loc[i + 1, "actual_price"]

        # Determine if we're going long or short
        if next_predicted_price - current_actual_price > limit:
            # Long position
            price_change = (
                next_actual_price - current_actual_price
            ) / current_actual_price
            # Calculate profit or loss with leverage
            profit_or_loss = current_amount * leverage * price_change

            print(profit_or_loss)

            # Update the current amount
            current_amount += profit_or_loss

        elif limit < current_actual_price - next_predicted_price:
            # Short position
            price_change = (
                current_actual_price - next_actual_price
            ) / current_actual_price

            # Calculate profit or loss with leverage
            profit_or_loss = current_amount * leverage * price_change

            print(profit_or_loss)

            # Update the current amount
            current_amount += profit_or_loss

    return current_amount


# Relative Strength Index (RSI)
def rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def features_engineering(df):
    # Assuming your data is in a DataFrame called `df`
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract useful numeric features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["year"] = df["timestamp"].dt.year

    # Columns for cyclical encoding
    cyclical_columns = ["hour", "day_of_week", "year"]

    # Cyclical encoding for hours (0-23)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Cyclical encoding for day_of_week (0-6, where 0 = Monday, 6 = Sunday)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Cyclical encoding for year (if applicable and cyclic)
    min_year = df["year"].min()
    max_year = df["year"].max()
    df["year_sin"] = np.sin(2 * np.pi * (df["year"] - min_year) / (max_year - min_year))
    df["year_cos"] = np.cos(2 * np.pi * (df["year"] - min_year) / (max_year - min_year))

    # Drop the original cyclical columns to avoid redundancy
    df = df.drop(columns=cyclical_columns)

    df["close_time"] = pd.to_datetime(df["close_time"])
    df = df.sort_values(by="timestamp")
    df = df.reset_index(drop=True)
    df = df.drop(columns=["close_time", "ignore", "timestamp"])
    df.isnull().sum()  # Identify missing values
    df = df.dropna()  # Drop rows with missing values (or use imputation if necessary)

    df["price_change"] = df["close"] - df["open"]
    df["price_change_pct"] = (df["price_change"] / df["open"]) * 100
    df["high_low_range"] = df["high"] - df["low"]

    df["volume_change_pct"] = df["volume"].pct_change()
    df["taker_buy_ratio"] = df["taker_buy_base_asset_volume"] / df["volume"]

    # Moving Average
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()

    df["rsi"] = rsi(df["close"])

    # Bollinger Bands
    df["bb_upper"] = df["ma_10"] + (df["close"].rolling(window=10).std() * 2)
    df["bb_lower"] = df["ma_10"] - (df["close"].rolling(window=10).std() * 2)

    for lag in range(1, 4):  # Create 1 to 3-step lag features
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(
        int
    )  # 1 if next close is higher

    df = df.dropna()

    # Normalization
    columns_to_normalize = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "price_change",
        "high_low_range",
        "ma_5",
        "ma_10",
        "rsi",
        "bb_upper",
        "bb_lower",
        "close_lag_1",
        "close_lag_2",
        "close_lag_3",
        "volume_lag_1",
        "volume_lag_2",
        "volume_lag_3",
        "quote_asset_volume",
        "taker_buy_quote_asset_volume",
        "taker_buy_base_asset_volume",
        "number_of_trades",
    ]

    # Normalize numerical features
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df
