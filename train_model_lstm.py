"""Train a univariate LSTM model for stock price prediction."""

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


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test_scaled: np.ndarray
    y_test_actual: np.ndarray
    scaler: MinMaxScaler


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


def build_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for start in range(len(data) - sequence_length):
        end = start + sequence_length
        X.append(data[start:end])
        y.append(data[end])

    X_array = np.asarray(X, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    return X_array, y_array


def prepare_dataset(stock_path: Path) -> Dataset:
    series = load_series(stock_path, TARGET_COLUMN)
    split_index = int(len(series) * SPLIT_RATIO)
    if split_index <= SEQUENCE_LENGTH:
        raise ValueError("Train split too small for the configured sequence length.")

    train_values = series[:split_index]
    test_values = series[split_index:]

    if len(test_values) <= SEQUENCE_LENGTH:
        raise ValueError("Test split too small for the configured sequence length.")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)

    X_train, y_train = build_sequences(train_scaled, SEQUENCE_LENGTH)
    X_test, y_test_scaled = build_sequences(test_scaled, SEQUENCE_LENGTH)

    # reshape to (samples, timesteps, features)
    X_train = X_train.reshape((-1, SEQUENCE_LENGTH, 1))
    X_test = X_test.reshape((-1, SEQUENCE_LENGTH, 1))

    y_test_actual = scaler.inverse_transform(y_test_scaled)

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test_scaled=y_test_scaled,
        y_test_actual=y_test_actual,
        scaler=scaler,
    )


def build_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    tf.random.set_seed(1234)
    np.random.seed(1234)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(50, activation="tanh", return_sequences=True),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.LSTM(30, activation="tanh", return_sequences=True),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.LSTM(20, activation="tanh"),
            tf.keras.layers.Dropout(0.01),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def train_once(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    model = build_model((SEQUENCE_LENGTH, 1))
    model.fit(dataset.X_train, dataset.y_train, epochs=EPOCHS, verbose=1)

    preds_scaled = model.predict(dataset.X_test, verbose=0)
    preds = dataset.scaler.inverse_transform(preds_scaled)
    return preds, dataset.y_test_actual


def evaluate(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float]:
    mae = mean_absolute_error(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions)
    accuracy = 1.0 - mape
    return float(mae), float(mape), float(accuracy)


def main() -> int:
    try:
        dataset = prepare_dataset(Path("stock_price.csv"))
    except Exception as exc:  # pragma: no cover - user-facing error message
        print(f"[ERROR] {exc}")
        return 1

    preds, actuals = train_once(dataset)
    mae, mape, accuracy = evaluate(preds, actuals)

    print("\n" + "=" * 60)
    print("LSTM MODEL RESULTS")
    print("=" * 60)
    print(f"Mean Absolute Error = ${mae:,.2f}")
    print(f"Mean Absolute Percentage Error = {mape * 100:.2f}%")
    print(f"Accuracy = {accuracy * 100:.2f}%")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
