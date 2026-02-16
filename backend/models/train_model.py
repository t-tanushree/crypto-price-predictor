import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from utils.data_pipeline import fetch_crypto_data, preprocess_data


# Directory to save trained models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def build_lstm_model(input_shape):
    """
    Build LSTM neural network model
    """

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


def train_and_save_model(symbol="BTC-USD"):
    """
    Train LSTM model and save it
    """

    print("Fetching latest data...")
    fetch_crypto_data(symbol)

    print("Preprocessing data...")
    X, y, scaler = preprocess_data(symbol)

    print("Building model...")
    model = build_lstm_model((X.shape[1], 1))

    print("Training model...")

    early_stop = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X,
        y,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop]
    )

    model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_model.h5")
    model.save(model_path)

    print(f"Model saved at {model_path}")

    return model_path


if __name__ == "__main__":
    train_and_save_model()
