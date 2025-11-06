"""
Assignment 4: Credit Card Fraud Detection using Autoencoder
Detecting fraudulent transactions
Dataset: Credit Card Fraud dataset (local)
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load Credit Card dataset
print("Loading Credit Card Fraud dataset...")
path = 'data/creditcardfraud-csv/creditcard.csv'
df = pd.read_csv(path)

print("\nDataset Information:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")

# Drop Time and Class columns
print("\nPreparing data...")
df = df.drop(['Time', 'Class'], axis=1)
print(f"After dropping Time and Class - shape: {df.shape}")

# Split into train and test sets
x_train, x_test = train_test_split(df, test_size=0.2, random_state=42)

print(f"\nTraining set shape: {x_train.shape}")
print(f"Testing set shape: {x_test.shape}")

# Build Autoencoder model
print("\nBuilding Autoencoder model...")

encoder = tf.keras.models.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(20, activation='relu')
])

decoder = tf.keras.models.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(x_train.shape[1], activation='linear')
])

model = tf.keras.models.Sequential([
    encoder,
    decoder
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

print("\nModel Summary:")
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    x_train,
    x_train,
    validation_data=(x_test, x_test),
    epochs=5,
    batch_size=100,
    shuffle=True
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Training History', size=20)
plt.xlabel('Epoch', size=15)
plt.ylabel('Loss (MSE)', size=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('creditcard_training_history.png')
print("Training history saved as 'creditcard_training_history.png'")
plt.show()

# Make predictions
print("\nDetecting anomalies...")
predictions = model.predict(x_test)

# Calculate reconstruction error (MSE)
mse = np.mean(np.power(x_test.values - predictions, 2), axis=1)

print(f"\nMSE Statistics:")
print(f"Mean: {np.mean(mse):.6f}")
print(f"Std: {np.std(mse):.6f}")
print(f"Min: {np.min(mse):.6f}")
print(f"Max: {np.max(mse):.6f}")

# Set threshold (95th percentile)
threshold = np.percentile(mse, 95)
print(f"\nAnomaly threshold (95th percentile): {threshold:.6f}")

# Identify anomalies
anomalies = mse > threshold
num_anomalies = np.sum(anomalies)
print(f"Number of anomalies detected: {num_anomalies}")
print(f"Percentage of anomalies: {(num_anomalies / len(mse)) * 100:.2f}%")

# Plot anomaly detection results
plt.figure(figsize=(14, 6))
plt.plot(mse, marker='o', linestyle='', markersize=3, label='MSE', alpha=0.6)
plt.axhline(threshold, color='r', linestyle='--', linewidth=2, label='Anomaly Threshold')
plt.xlabel('Sample Index', size=12)
plt.ylabel('Mean Squared Error', size=12)
plt.title('Credit Card Fraud Detection - Anomaly Detection Results', size=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('creditcard_anomaly_detection.png')
print("Anomaly detection plot saved as 'creditcard_anomaly_detection.png'")
plt.show()

# Plot reconstruction comparison for first sample
plt.figure(figsize=(14, 6))
plt.plot(x_test.iloc[0].values, label='Original Transaction', linewidth=2, marker='o')
plt.plot(predictions[0], label='Reconstructed Transaction', linewidth=2, marker='x')
plt.xlabel('Feature Index', size=12)
plt.ylabel('Feature Value', size=12)
plt.title('Sample Transaction Reconstruction', size=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('creditcard_reconstruction_sample.png')
print("Sample reconstruction saved as 'creditcard_reconstruction_sample.png'")
plt.show()

# Confusion matrix (comparing anomalies with themselves)
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(6, 4.75))
cm = confusion_matrix(anomalies, anomalies)
sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.title("Confusion Matrix", fontsize=14)
plt.savefig('creditcard_confusion_matrix.png')
print("Confusion matrix saved as 'creditcard_confusion_matrix.png'")
plt.show()

# MSE distribution histogram
plt.figure(figsize=(10, 6))
plt.hist(mse, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Mean Squared Error', size=12)
plt.ylabel('Frequency', size=12)
plt.title('Distribution of Reconstruction Errors', size=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('creditcard_mse_distribution.png')
print("MSE distribution saved as 'creditcard_mse_distribution.png'")
plt.show()

print("\nAssignment 4 (Credit Card Fraud Detection) completed successfully!")
print(f"\nSummary:")
print(f"Total transactions analyzed: {len(x_test)}")
print(f"Fraudulent transactions detected: {num_anomalies}")
print(f"Fraud rate: {(num_anomalies / len(mse)) * 100:.2f}%")
