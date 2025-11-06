"""
Assignment 3: Image Classification with CIFAR-10 Dataset
Building image classification model using CNN
Dataset: CIFAR-10 (10 object classes)
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Display a sample image
plt.figure(figsize=(5, 5))
plt.imshow(x_train[25])
plt.title(f"Class: {class_names[y_train[25][0]]}", size=15)
plt.axis('off')
plt.savefig('cifar10_sample_image.png')
print("Sample image saved as 'cifar10_sample_image.png'")
plt.show()

# Scale the values to [0, 1]
print("\nScaling pixel values...")
x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"After scaling - x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

# Convert labels to categorical
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"y_train shape after conversion: {y_train.shape}")
print(f"y_test shape after conversion: {y_test.shape}")

# Build CNN model
print("\nBuilding CNN model...")
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
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
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy', size=25, pad=20)
plt.ylabel('Accuracy', size=15)
plt.xlabel('Epoch', size=15)
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss', size=25, pad=20)
plt.ylabel('Loss', size=15)
plt.xlabel('Epoch', size=15)
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.savefig('cifar10_training_history.png')
print("Training history plot saved as 'cifar10_training_history.png'")
plt.show()

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(x_test)

# Display sample predictions
for sample_index in [0, 100]:
    plt.figure(figsize=(5, 5))
    plt.imshow(x_test[sample_index].reshape(32, 32, -1))
    predicted_class = class_names[np.argmax(predictions[sample_index])]
    plt.title(f"Predicted: {predicted_class}", size=15)
    plt.axis('off')
    plt.savefig(f'cifar10_prediction_{sample_index}.png')
    print(f"Prediction plot saved as 'cifar10_prediction_{sample_index}.png'")
    plt.show()
    print(f"Sample {sample_index} - Predicted: {predicted_class}")

print("\nAssignment 3 (CIFAR-10) completed successfully!")
