import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from src.config import data_config
from src.data import plot, retriever

symbol_list = ["BTCUSDT", "ETHUSDT"]
my_interval = data_config.DEFAULT_INTERVAL
my_start_date = "2017-01-01"
filename_format = data_config.GENERAL_FILENAME
retriever.save_data(symbol_list, my_interval, my_start_date)
retriever.update_data("BTCUSDT")
retriever.update_data("ETHUSDT")
btc = retriever.retrieve_data("BTCUSDT")
eth = retriever.retrieve_data("ETHUSDT")
missing_btc = retriever.find_missing_data(btc, my_interval)
missing_eth = retriever.find_missing_data(eth, my_interval)

plot.plot(btc[["open", "high", "low", "close"]], True)

btc_daily = retriever.aggregate_data(btc, "1d")
