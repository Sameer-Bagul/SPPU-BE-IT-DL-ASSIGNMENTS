"""
Assignment 2: Feedforward Neural Network with MNIST Dataset
Implementing feedforward neural network using Keras and TensorFlow
Dataset: MNIST (Handwritten Digits)
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

# Load MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Display dataset information
print(f"Shape of X_train: {x_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of x_test: {x_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Display unique labels
print(f"Unique training labels: {np.unique(y_train)}")
print(f"Unique testing labels: {np.unique(y_test)}")

# Scale the values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"After scaling - x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

# Build Neural Network
print("\nBuilding Neural Network...")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation='relu', name='L1'),
    keras.layers.Dense(50, activation='relu', name='L2'),
    keras.layers.Dense(10, activation='softmax', name='L3')
])

# Compile the model
model.compile(
    optimizer="sgd",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    x_train, y_train,
    batch_size=30,
    epochs=10,
    validation_data=(x_test, y_test),
    shuffle=True
)

# Plot training history
plt.figure(figsize=[15, 8])

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy', size=25, pad=20)
plt.ylabel('Accuracy', size=15)
plt.xlabel('Epoch', size=15)
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss', size=25, pad=20)
plt.ylabel('Loss', size=15)
plt.xlabel('Epoch', size=15)
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.savefig('mnist_training_history.png')
print("Training history plot saved as 'mnist_training_history.png'")
plt.show()

# Evaluate the model
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions
print("\nMaking predictions...")
predicted_value = model.predict(x_test)

# Display a sample prediction
sample_index = 15
plt.figure(figsize=(5, 5))
plt.imshow(x_test[sample_index], cmap='gray')
plt.title(f"Predicted: {np.argmax(predicted_value[sample_index])}", size=15)
plt.axis('off')
plt.savefig('mnist_sample_prediction.png')
print(f"Sample prediction plot saved as 'mnist_sample_prediction.png'")
plt.show()

print(f"\nPredicted digit for sample {sample_index}: {np.argmax(predicted_value[sample_index])}")
print(f"Actual digit: {y_test[sample_index]}")

print("\nAssignment 2 (MNIST) completed successfully!")
