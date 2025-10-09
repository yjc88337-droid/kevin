"""Train the FinBERT-enhanced LSTM model using price and sentiment features."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler


SPLIT_RATIO: float = 0.85
SEQUENCE_LENGTH: int = 10
EPOCHS: int = 100
LEARNING_RATE: float = 0.02
TARGET_COLUMN: str = "Close"
SENTIMENT_COLUMN: str = "FinBERT score"


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test_scaled: np.ndarray
    y_test_actual: np.ndarray
    price_scaler: MinMaxScaler
    sentiment_scaler: MinMaxScaler


def load_series(csv_path: Path, column: str) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"Required data file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if column not in frame.columns:
        raise ValueError(f"Column '{column}' missing from {csv_path}")

    values = frame[column].astype("float32").to_numpy().reshape(-1, 1)
    if values.shape[0] <= SEQUENCE_LENGTH:
        raise ValueError("Not enough rows to create training sequences.")
    return values


def build_multivariate_sequences(
    price_data: np.ndarray,
    sentiment_data: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for start in range(len(price_data) - sequence_length):
        end = start + sequence_length
        price_window = price_data[start:end]
        sentiment_window = sentiment_data[start:end]
        features = np.concatenate([price_window, sentiment_window], axis=1)
        X.append(features)
        y.append(price_data[end])

    X_array = np.asarray(X, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    return X_array, y_array


def prepare_dataset(stock_path: Path, sentiment_path: Path) -> Dataset:
    prices = load_series(stock_path, TARGET_COLUMN)
    sentiments = load_series(sentiment_path, SENTIMENT_COLUMN)

    if prices.shape[0] != sentiments.shape[0]:
        raise ValueError(
            "stock_price.csv and sentiment.csv must contain the same number of rows."
        )

    split_index = int(len(prices) * SPLIT_RATIO)
    if split_index <= SEQUENCE_LENGTH:
        raise ValueError("Train split too small for the configured sequence length.")

    price_train, price_test = prices[:split_index], prices[split_index:]
    sentiment_train, sentiment_test = sentiments[:split_index], sentiments[split_index:]

    if len(price_test) <= SEQUENCE_LENGTH:
        raise ValueError("Test split too small for the configured sequence length.")

    price_scaler = MinMaxScaler()
    sentiment_scaler = MinMaxScaler(feature_range=(-1, 1))

    price_train_scaled = price_scaler.fit_transform(price_train)
    price_test_scaled = price_scaler.transform(price_test)

    sentiment_train_scaled = sentiment_scaler.fit_transform(sentiment_train)
    sentiment_test_scaled = sentiment_scaler.transform(sentiment_test)

    X_train, y_train = build_multivariate_sequences(
        price_train_scaled, sentiment_train_scaled, SEQUENCE_LENGTH
    )
    X_test, y_test_scaled = build_multivariate_sequences(
        price_test_scaled, sentiment_test_scaled, SEQUENCE_LENGTH
    )

    y_test_actual = price_scaler.inverse_transform(y_test_scaled)

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test_scaled=y_test_scaled,
        y_test_actual=y_test_actual,
        price_scaler=price_scaler,
        sentiment_scaler=sentiment_scaler,
    )


def build_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    tf.random.set_seed(1234)
    np.random.seed(1234)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(70, activation="tanh", return_sequences=True),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.LSTM(30, activation="tanh", return_sequences=True),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.LSTM(10, activation="tanh"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def train_once(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    model = build_model((SEQUENCE_LENGTH, dataset.X_train.shape[2]))
    model.fit(dataset.X_train, dataset.y_train, epochs=EPOCHS, verbose=1)

    preds_scaled = model.predict(dataset.X_test, verbose=0)
    preds = dataset.price_scaler.inverse_transform(preds_scaled)
    return preds, dataset.y_test_actual


def evaluate(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float]:
    mae = mean_absolute_error(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions)
    accuracy = 1.0 - mape
    return float(mae), float(mape), float(accuracy)


def main() -> int:
    try:
        dataset = prepare_dataset(Path("stock_price.csv"), Path("sentiment.csv"))
    except Exception as exc:  # pragma: no cover - user-facing error message
        print(f"[ERROR] {exc}")
        return 1

    preds, actuals = train_once(dataset)
    mae, mape, accuracy = evaluate(preds, actuals)

    print("\n" + "=" * 60)
    print("FINBERT-LSTM MODEL RESULTS")
    print("=" * 60)
    print(f"Mean Absolute Error = ${mae:,.2f}")
    print(f"Mean Absolute Percentage Error = {mape * 100:.2f}%")
    print(f"Accuracy = {accuracy * 100:.2f}%")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
