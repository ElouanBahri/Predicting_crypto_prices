import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.decomposition import PCA

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.data import binance, plot, retriever
from src.notebooks.modules.utils import feature_engineering_last


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

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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

        if self.preprocessed_data is None or self.data.empty:
            print("No preprocessed data available to make prediction.")
            return

        # Step 1: Extract most recent window of features
        # You must adapt this based on how your model was trained (e.g., sequence length)
        sequence_length = self.rnn_model.input_shape[
            1
        ]  # e.g., 60 if you trained on 60 time steps

        df = self.preprocessed_data.copy()
        X_features = df.drop(
            columns=["target"], errors="ignore"
        )  # drop target if present

        if len(df) < sequence_length:
            print(
                f"Not enough data to make prediction. Need at least {sequence_length} rows."
            )
            return

        latest_sequence = df.iloc[-sequence_length:].values
        latest_sequence = df.reshape(1, sequence_length, -1)  # reshape for RNN input

        # Step 2: Predict next return
        predicted_return = self.rnn_model.predict(latest_sequence)[0][0]
        print(f"Predicted next return: {predicted_return:.5f}")

        return predicted_return
