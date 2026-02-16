import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Directory to store data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_crypto_data(symbol="BTC-USD", period="2y", interval="1d"):
    """
    Fetch historical crypto data using yfinance
    and save it as CSV.
    """
    print(f"Fetching data for {symbol}...")
    df = yf.download(symbol, period=period, interval=interval)

    if df.empty:
        raise ValueError("No data fetched. Check symbol or internet connection.")

    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df.to_csv(file_path)

    print(f"Data saved to {file_path}")
    return df


def preprocess_data(symbol="BTC-USD", sequence_length=60):
    """
    Load CSV, normalize Close prices,
    and create sequences for LSTM training.
    """

    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Data file not found. Run fetch_crypto_data() first."
        )

    # Load CSV
    df = pd.read_csv(file_path)

    # Clean Close column
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    # Reshape for scaler
    close_prices = df["Close"].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create sequences
    X = []
    y = []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape for LSTM (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    print(f"Preprocessed data: X shape = {X.shape}, y shape = {y.shape}")

    return X, y, scaler
