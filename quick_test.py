# test_model_quick.py - Quick model test with reduced epochs
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf

print("[INFO] Testing MLP model with reduced epochs...")

# Reduced hyperparameters for quick test
split = 0.85
sequence_length = 10
epochs = 5  # Reduced from 100
learning_rate = 0.01

# Loading stock price data
stock_data = pd.read_csv("stock_price.csv")
column = ['Close']

len_stock_data = stock_data.shape[0]
print(f"[INFO] Loaded {len_stock_data} rows of stock data")

# Splitting data to train and test
train_examples = int(len_stock_data * split)
train = stock_data.get(column).values[:train_examples]
test = stock_data.get(column).values[train_examples:]
len_train = train.shape[0]
len_test = test.shape[0]

print(f"[INFO] Train: {len_train} rows, Test: {len_test} rows")

# Normalizing data
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

# Splitting training data to x and y
X_train = []
for i in range(len_train - sequence_length):
    X_train.append(train[i : i + sequence_length])
X_train = np.array(X_train, dtype=float).reshape(-1, sequence_length)
y_train = np.array(train[sequence_length:], dtype=float).reshape(-1, 1)

# Splitting testing data to x and y
X_test = []
for i in range(len_test - sequence_length):
    X_test.append(test[i : i + sequence_length])
X_test = np.array(X_test, dtype=float).reshape(-1, sequence_length)
y_test = np.array(test[sequence_length:], dtype=float).reshape(-1, 1)

print(f"[INFO] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"[INFO] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Creating MLP model
print("[INFO] Building MLP model...")
tf.random.set_seed(1234)
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(units=50, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=30, activation="relu"),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(units=20, activation="relu"),
    tf.keras.layers.Dropout(0.01),
    tf.keras.layers.Dense(units=1, activation="linear")
])

model.compile(
    loss='mse',  # Updated for TensorFlow 2.x
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
)

print("[INFO] Training model...")
model.fit(X_train, y_train, epochs=epochs, verbose=1)

# Inverting normalization
y_test = scaler.inverse_transform(y_test)

# Prediction on test set
print("[INFO] Making predictions...")
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Evaluation
mae = mean_absolute_error(predictions, y_test)
mape = mean_absolute_percentage_error(predictions, y_test)
accuracy = 1 - mape

print("\n" + "="*50)
print("QUICK TEST RESULTS (5 epochs)")
print("="*50)
print(f"Mean Absolute Error = ${mae:.2f}")
print(f"Mean Absolute Percentage Error = {mape*100:.2f}%")
print(f"Accuracy = {accuracy*100:.2f}%")
print("="*50)
print("\n[SUCCESS] Model training pipeline works!")
print("[INFO] You can now run the full models with 100 epochs")
