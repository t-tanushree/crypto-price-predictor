import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "BTC-USD_lstm_model.h5"
)

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "BTC-USD.csv"
)


def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def predict_next_price():

    # Load CSV
    df = pd.read_csv(DATA_PATH)

    # 🔥 Important Fix: Ensure Close column is numeric
    df = df[["Close"]]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Drop any bad rows
    df = df.dropna()

    close_prices = df["Close"].values.reshape(-1, 1)

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Last 60 days
    last_60_days = scaled_data[-60:]

    X_test = np.reshape(last_60_days, (1, 60, 1))

    model = load_model()

    predicted_scaled = model.predict(X_test)

    predicted_price = scaler.inverse_transform(predicted_scaled)

    return float(predicted_price[0][0])
