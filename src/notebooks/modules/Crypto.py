import pandas as pd
import requests
import time
import os
import sys
from src.data import plot, retriever
from datetime import datetime
from sklearn.decomposition import PCA
import numpy as np


# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data import binance
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
        self.data = None  # Placeholder for historical data
        
        # Fetch initial data
        self.update_data()

    def fetch_data(self, start_time=None):
        """
        Fetch historical Kline (candlestick) data from Binance API.
        :param start_time: Optional, fetch data from a specific timestamp
        """
        my_start_date = start_time

        self.data = binance.get_binance_data(self.symbol, self.interval, my_start_date)
        


    def update_data(self):

        current_data = self
        last_date = current_data.index[-1]
        last_date_str = datetime.strftime(last_date, "%Y-%m-%d")
        new_data = binance.get_binance_data(self.symbol, self.interval, last_date_str)
        all_data = pd.concat([current_data, new_data], axis=0)
        all_data = all_data[~all_data.index.duplicated(keep="last")]
        all_data.sort_index(inplace=True)
        self.data = all_data

    def preprocess_data(self):
        """Preprocess the data (e.g., adding indicators, normalization)."""
        if self.data is None:
            print("No data available for preprocessing.")
            return
        
        df = self.data.copy()
        
        # Example: Compute moving averages
        df = feature_engineering_last(df)
        y = df['target']

        df_without_target= df.drop(columns=['target'], errors='ignore')  # Only keep features
        

        explained_variance = np.cumsum(pca.explained_variance_ratio_)

        num_components = np.argmax(explained_variance >= 0.95) + 1
        print(f"Optimal number of components: {num_components}")

        # Transform data using optimal number of components
        pca = PCA(n_components=num_components)
        X_reduced = pca.fit_transform(df_without_target)

        # Convert back to DataFrame
        columns = [f'PC{i+1}' for i in range(num_components)]
        X_pca_df = pd.DataFrame(X_reduced, columns=columns)

        self.data = df  # Update the class attribute with preprocessed data
        print("Data preprocessing complete.")

    def show_data(self, rows=5):
        """Display the latest data."""
        if self.data is not None:
            print(self.data.tail(rows))
        else:
            print("No data available.")