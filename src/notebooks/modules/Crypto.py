import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.data import binance, plot, retriever
from src.notebooks.modules.utils import (
    feature_engineering_last,
    filter_data_by_year_month,
)


class CryptoData:
    def __init__(self, symbol: str, interval: str):
        """
        Initialize with crypto symbol and time interval.
        :param symbol: Cryptocurrency pair, e.g., 'BTCUSDT'
        :param interval: Timeframe, e.g., '1h', '1d'
        """
        self.symbol = symbol.upper()
        self.interval = interval
        self.raw_data = None  # Placeholder for historical data
        self.preprocessed_data = None
        self.rnn_model = None

    def fetch_data(self, start_time=None):
        """
        Fetch historical Kline (candlestick) data from Binance API.
        :param start_time: Optional, fetch data from a specific timestamp
        """
        my_start_date = start_time

        self.raw_data = binance.get_binance_data(
            self.symbol, self.interval, my_start_date
        )

    def update_data(self):
        current_data = self.raw_data
        last_date = current_data.index[-1]
        last_date_str = datetime.strftime(last_date, "%Y-%m-%d")
        new_data = binance.get_binance_data(self.symbol, self.interval, last_date_str)
        all_data = pd.concat([current_data, new_data], axis=0)
        all_data = all_data[~all_data.index.duplicated(keep="last")]
        all_data.sort_index(inplace=True)
        self.raw_data = all_data

    def preprocess_data(self):
        """Preprocess the data (e.g., adding indicators, normalization)."""
        if self.raw_data is None:
            print("No data available for preprocessing.")
            return

        df = self.raw_data.copy()

        YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

        df = filter_data_by_year_month(df, YEARS)

        df = feature_engineering_last(df)
        y = df["target"]

        df_without_target = df.drop(
            columns=["target"], errors="ignore"
        )  # Only keep features

        pca = PCA()
        pca.fit(df_without_target)

        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = (
            np.argmax(explained_variance >= 0.95) + 1
        )  # Find min components for 95% variance

        if num_components == 0:
            print(
                "⚠️ No PCA components meet 95% explained variance. Using 1 component."
            )
            num_components = 1

        print(f"✅ Optimal number of components: {num_components}")

        # ✅ Step 2: Fit PCA with optimal number of components
        pca = PCA(n_components=num_components)
        X_reduced = pca.fit_transform(df_without_target)

        # ✅ Convert back to DataFrame and restore index
        columns = [f"PC{i+1}" for i in range(num_components)]
        X_pca_df = pd.DataFrame(
            X_reduced, columns=columns, index=df.index
        )  # Keep original index

        X_pca_df["target"] = y  # Add target column back

        self.preprocessed_data = (
            X_pca_df  # Update the class attribute with preprocessed data
        )
        print("Data preprocessing complete.")

    def show_raw_data(self):
        """Display the latest data."""
        if self.raw_data is not None:
            print(self.raw_data.iloc[[0, -1]])
            print("------------------------------------------------")
            print("list of features :")
            print(self.raw_data.columns)
            print("shape: ")
            print(self.raw_data.shape)
        else:
            print("No data available.")

    def load_model(self, path=""):
        self.rnn_model = tf.keras.models.load_model(path)
        print("You've uploaded this model :")
        self.rnn_model.summary()

    def make_prediction(self):
        if self.rnn_model is None:
            print("Model not loaded. Please call load_model() first.")
            return

        if self.preprocessed_data is None or self.preprocessed_data.empty:
            print("No preprocessed data available to make prediction.")
            return

        # Get expected sequence length and number of features
        sequence_length = self.rnn_model.input_shape[1]
        num_features = self.rnn_model.input_shape[2]

        df = self.preprocessed_data.copy()

        X_features = df.drop(columns=["target"], errors="ignore")

        if len(X_features) < sequence_length:
            print(
                f"❌ Not enough data. Required: {sequence_length}, Found: {len(X_features)}"
            )
            return

        # Extract the latest sequence
        latest_sequence = X_features.iloc[-sequence_length:].to_numpy()

        if latest_sequence.shape[1] != num_features:
            print(
                f"❌ Feature mismatch: Model expects {num_features} features, but found {latest_sequence.shape[1]}"
            )
            return

        # Reshape for model: (1, time_steps, features)
        latest_sequence = latest_sequence.reshape(1, sequence_length, num_features)

        # Predict
        proba_up = self.rnn_model.predict(latest_sequence, verbose=0)[0][0]

        print(f"proba that it's going up  : {proba_up}")
        if proba_up > 0.55:
            return "Buy"
        elif proba_up < 0.44:
            return "Sell"
        else:
            return "Hold"

    def backtesting(
        self, threshold_buy=0.55, threshold_sell=0.44, initial_cash=10000
    ):
        if self.rnn_model is None:
            print("❌ Model not loaded.")
            return
        if self.preprocessed_data is None:
            print("❌ Data not preprocessed.")
            return

        sequence_length = self.rnn_model.input_shape[1]
        num_features = self.rnn_model.input_shape[2]

        X = self.preprocessed_data.drop(columns=["target"]).values  # Features
        y = self.preprocessed_data["target"].values  # Target variable


        df_real = self.raw_data.copy()
        YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
        df_real = filter_data_by_year_month(df_real, YEARS)
        df_real = feature_engineering_last(df_real)
        returns = df_real["returns", ""]
                

        # Reshape the data into sequences (timesteps)
        timesteps = sequence_length  # Number of timesteps for the RNN
        X_sequences = []
        y_sequences = []
        returns_sequences = []

        for i in range(len(X) - timesteps):
            X_sequences.append(X[i : i + timesteps])
            y_sequences.append(y[i + timesteps])
            returns_sequences.append(returns[i + timesteps])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        returns_sequences = np.array(returns_sequences)

        split_index = int(len(X_sequences) * 0.8)  # 80% train, 20% test
        X_train, X_test = X_sequences[:split_index], X_sequences[split_index:]
        y_train, y_test = y_sequences[:split_index], y_sequences[split_index:]
        returns_sequences_test = returns_sequences[split_index:]

        positions = []
        cash = initial_cash
        holdings = 0
        capital = [initial_cash]

        y_pred_proba = self.rnn_model.predict(X_test)[:, 0]  # Assuming sigmoid output
       

        # Loop over each prediction and return
        for i in range(len(y_pred_proba)):
            if y_pred_proba[i] > threshold_buy:  # Buy
                capital.append(capital[-1] * (1 + returns_sequences_test[i]))
            
            elif y_pred_proba[i] < threshold_sell : #Short
                
                capital.append(capital[-1] * (1 - returns_sequences_test[i]))

            else:  # Hold in cash
                capital.append(capital[-1])  # no gain/loss

        capital = np.array(capital[1:])  # Drop initial entry

        daily_returns = np.diff(capital) / capital[:-1]
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

        

        #accuracy = accuracy_score(y_test, y_pred)
        #print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"📈 Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"💰 Final Capital: ${capital[-1]:,.2f}")

        plt.figure(figsize=(12, 5))
        plt.plot(capital, label="Equity Curve")
        plt.title("Backtest Equity Curve")
        plt.xlabel("Days")
        plt.ylabel("Capital ($)")
        plt.grid(True)
        plt.legend()
        plt.show()