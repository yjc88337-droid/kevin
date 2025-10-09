"""Train a feed-forward neural network baseline for stock price prediction.

This script consumes ``stock_price.csv`` produced by the data collection step,
splits it into train/test sets, fits a simple multi-layer perceptron, and
reports common regression metrics.  The goal is to provide a reliable building
block for the full pipeline so the training run can be executed end-to-end
without manual tweaking.
"""

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPLIT_RATIO: float = 0.85
SEQUENCE_LENGTH: int = 10
EPOCHS: int = 100
LEARNING_RATE: float = 0.01
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
    """Load a univariate time-series column as a 2D ``np.ndarray``."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Required data file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if column not in data.columns:
        raise ValueError(f"Column '{column}' missing from {csv_path}")

    values = data[column].astype("float32").to_numpy().reshape(-1, 1)
    if values.shape[0] <= SEQUENCE_LENGTH:
        raise ValueError(
            "Not enough rows in the dataset to create training sequences."
        )
    return values


def build_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows (X, y) from a 2D time-series array."""

    X, y = [], []
    for start in range(len(data) - sequence_length):
        end = start + sequence_length
        X.append(data[start:end])
        y.append(data[end])

    return np.asarray(X, dtype=np.float32).reshape(-1, sequence_length), np.asarray(y, dtype=np.float32)


def prepare_dataset(stock_path: Path) -> Dataset:
    """Load data, split into train/test, scale, and prepare sequences."""

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

    # y_test_actual is in the original scale for metric computation
    y_test_actual = scaler.inverse_transform(y_test_scaled)

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test_scaled=y_test_scaled,
        y_test_actual=y_test_actual,
        scaler=scaler,
    )


def build_model(input_dim: int) -> tf.keras.Model:
    """Construct the baseline MLP architecture."""

    tf.keras.backend.clear_session()
    tf.random.set_seed(1234)
    np.random.seed(1234)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(30, activation="relu"),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(20, activation="relu"),
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
    """Train the model once and return predictions + actual targets."""

    model = build_model(dataset.X_train.shape[1])
    model.fit(dataset.X_train, dataset.y_train, epochs=EPOCHS, verbose=1)

    preds_scaled = model.predict(dataset.X_test, verbose=0)
    preds = dataset.scaler.inverse_transform(preds_scaled)
    return preds, dataset.y_test_actual


def evaluate(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float]:
    """Compute MAE, MAPE, and accuracy (1 - MAPE)."""

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
    print("MLP MODEL RESULTS")
    print("=" * 60)
    print(f"Mean Absolute Error = ${mae:,.2f}")
    print(f"Mean Absolute Percentage Error = {mape * 100:.2f}%")
    print(f"Accuracy = {accuracy * 100:.2f}%")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
