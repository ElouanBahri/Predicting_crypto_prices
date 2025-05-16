
# 🧠 Crypto Price Prediction & Algorithmic Trading

This project explores multiple methods for forecasting cryptocurrency returns and building automated trading strategies. It combines classical time series models, machine learning, and large language models (LLMs) to evaluate market behavior and design profitable strategies.

## 🔍 Project Objectives

- Predict short-term cryptocurrency price movements (15-min candles)
- Compare classical (ARIMA, GARCH), ML (LSTM, XGBoost), and LLM-based (TimeGPT, Ollama) methods
- Build and backtest trading strategies based on these predictions
- Generalize the pipeline to multiple cryptocurrencies using OOP

## 🧰 Tech Stack

- **Python** (OOP structure with a `CryptoData` class)
- **Machine Learning**: TensorFlow/Keras, XGBoost
- **Time Series Models**: statsmodels (ARIMA), arch (GARCH)
- **LLMs**: TimeGPT (via Nixtla API), LLaMA (via Ollama)
- **APIs**: Binance for historical data, CryptoPanic for news

## 📈 Features

- Feature engineering (RSI, MACD, Bollinger Bands, etc.)
- Rolling-window backtesting with equity curve and Sharpe ratio
- Real-time prediction module (cron job compatible)
- Sentiment analysis from news headlines using local LLM

## 📄 Results Summary

- LSTM strategy: Accuracy ≈ 56%, Sharpe ≈ 1.07
- TimeGPT: Accuracy ≈ 48%, Sharpe ≈ 1.27 (on sample)
- Final capital (backtest 2024): ~$12,500 from $10,000

 

This project uses **Python == 3.11**.

## 1. Installation

### 1.1. Virtual environment
```bash
conda env create -f src/environment/conda_dependencies.yml
conda activate Predicting_crypto
```

### 1.2. Dev guidelines

1. To update your environment, make sure to run :
```bash
pip install -r src/environment/requirements.txt
```

2. To format your code, you can run :
```bash
invoke format
```

