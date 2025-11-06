"""
Assignment 4: Anomaly Detection using Autoencoder
Detecting anomalies in ECG signals
Dataset: ECG time-series data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Load ECG dataset
print("Loading ECG dataset...")
path = 'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'
data = pd.read_csv(path, header=None)

print("\nDataset Information:")
print(data.head())
print(f"\nDataset shape: {data.shape}")
data.info()

# Split features and target
print("\nPreparing data...")
features = data.drop(140, axis=1)  # All columns except last
target = data[140]  # Last column (labels)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Select only normal data for training (labeled as 1)
train_index = y_train[y_train == 1].index
train_data = x_train.loc[train_index]

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {x_test.shape}")

# Scale the data
print("\nScaling data...")
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = min_max_scaler.fit_transform(train_data.copy())
x_test_scaled = min_max_scaler.transform(x_test.copy())

# Build Autoencoder model
print("\nBuilding Autoencoder model...")

class AutoEncoder(Model):
    def __init__(self, output_units, ldim=8):
        super().__init__()
        # Encoder
        self.encoder = Sequential([
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(ldim, activation='relu')
        ])
        # Decoder
        self.decoder = Sequential([
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(output_units, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Create model instance
model = AutoEncoder(output_units=x_train_scaled.shape[1])

# Compile the model
model.compile(loss='msle', metrics=['mse'], optimizer='adam')

print("\nModel created successfully!")

# Train the model
print("\nTraining the model...")
history = model.fit(
    x_train_scaled,
    x_train_scaled,
    epochs=20,
    batch_size=512,
    validation_data=(x_test_scaled, x_test_scaled),
    shuffle=True
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title('Model Loss', size=20)
plt.xlabel('Epoch', size=15)
plt.ylabel('Loss', size=15)
plt.legend()
plt.grid(True)
plt.savefig('ecg_training_history.png')
print("Training history saved as 'ecg_training_history.png'")
plt.show()

# Find threshold for anomaly detection
print("\nFinding anomaly threshold...")

def find_threshold(model, x_train_scaled):
    recons = model.predict(x_train_scaled)
    recons_error = tf.keras.metrics.msle(recons, x_train_scaled)
    threshold = np.mean(recons_error.numpy()) + np.std(recons_error.numpy())
    return threshold

threshold = find_threshold(model, x_train_scaled)
print(f"Threshold: {threshold:.6f}")

# Make predictions
print("\nDetecting anomalies...")

def get_predictions(model, x_test_scaled, threshold):
    predictions = model.predict(x_test_scaled)
    errors = tf.keras.losses.msle(predictions, x_test_scaled)
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
    return preds

predictions = get_predictions(model, x_test_scaled, threshold)

# Calculate accuracy
accuracy = accuracy_score(predictions, y_test)
print(f"Accuracy Score: {accuracy:.4f}")

# Visualize results
print("\nGenerating visualizations...")

# Reconstruct test data
reconstructions = model.predict(x_test_scaled)
errors = tf.keras.losses.msle(reconstructions, x_test_scaled)

# Plot reconstruction error
plt.figure(figsize=(12, 6))
plt.plot(errors.numpy(), marker='o', linestyle='', markersize=3, label='Reconstruction Error')
plt.axhline(threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Sample Index', size=12)
plt.ylabel('Reconstruction Error (MSLE)', size=12)
plt.title('ECG Anomaly Detection Results', size=15)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ecg_anomaly_detection.png')
print("Anomaly detection plot saved as 'ecg_anomaly_detection.png'")
plt.show()

# Plot sample ECG signals
plt.figure(figsize=(12, 6))
plt.plot(x_test_scaled[0], label='Original ECG', linewidth=2)
plt.plot(reconstructions[0], label='Reconstructed ECG', linewidth=2)
plt.xlabel('Time', size=12)
plt.ylabel('Amplitude', size=12)
plt.title('Normal ECG Signal Reconstruction', size=15)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ecg_reconstruction_sample.png')
print("Sample reconstruction saved as 'ecg_reconstruction_sample.png'")
plt.show()

print("\nAssignment 4 (ECG Anomaly Detection) completed successfully!")
print(f"Total anomalies detected: {len(predictions[predictions == 0.0])}")
print(f"Total normal samples: {len(predictions[predictions == 1.0])}")
