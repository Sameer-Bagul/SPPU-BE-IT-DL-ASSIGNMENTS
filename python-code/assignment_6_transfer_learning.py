"""
Assignment 6: Transfer Learning with VGG16
Object detection using pre-trained CNN architecture
Dataset: Caltech-101 (local dataset)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 70)
print("Assignment 6: Transfer Learning with VGG16")
print("=" * 70)

# Dataset configuration
dataset_dir = 'data/caltech-101-img'
weights_path = 'data/vgg/vgg16_weights_tf_dim_ordering_tf_kernels_notop (1).h5'

# Check if paths exist
print("\nChecking dataset and weights...")
print(f"Dataset directory exists: {os.path.exists(dataset_dir)}")
print(f"VGG16 weights exist: {os.path.exists(weights_path)}")

# Load and preprocess data
print("\n" + "=" * 70)
print("Loading and preprocessing data...")
print("=" * 70)

dataset_datagen = ImageDataGenerator(rescale=1.0 / 255)

batch_size = 2000
dataset_generator = dataset_datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical'
)

print(f"\nTotal classes found: {len(dataset_generator.class_indices)}")
print(f"Batch size: {batch_size}")

# Get train and test data
x_train, y_train = dataset_generator[0]
x_test, y_test = dataset_generator[1]

print(f"\nTraining set size: {len(x_train)}")
print(f"Testing set size: {len(x_test)}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")
print(f"Image shape: {x_train[0].shape}")

# Get class labels
labels = list(dataset_generator.class_indices.keys())
print(f"\nSample classes: {labels[:10]}")

# Display sample images
print("\nGenerating sample images visualization...")
plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_train[i])
    class_idx = np.argmax(y_train[i])
    plt.title(f"{labels[class_idx]}", size=10)
    plt.axis('off')
plt.tight_layout()
plt.savefig('caltech101_sample_images.png')
print("Sample images saved as 'caltech101_sample_images.png'")
plt.show()

# Stage 1: Transfer Learning with Frozen Layers
print("\n" + "=" * 70)
print("STAGE 1: Transfer Learning with All Layers Frozen")
print("=" * 70)

# Load VGG16 base model
base_model = VGG16(weights=weights_path, include_top=False, input_shape=(64, 64, 3))

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False

print(f"\nBase model layers: {len(base_model.layers)}")
print("All layers frozen for initial training")

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(64, activation='relu')(x)
predictions = Dense(102, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

# Train the model
print("\nTraining stage 1...")
history1 = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    validation_data=(x_test, y_test)
)

# Evaluate stage 1
loss1, acc1 = model.evaluate(x_test, y_test)
print(f"\nStage 1 Results:")
print(f"  Loss: {loss1:.4f}")
print(f"  Accuracy: {acc1:.4f} ({acc1*100:.2f}%)")

# Stage 2: Fine-tuning with Unfrozen Layers
print("\n" + "=" * 70)
print("STAGE 2: Fine-tuning with Partially Unfrozen Layers")
print("=" * 70)

# Load fresh base model
base_model = VGG16(weights=weights_path, include_top=False, input_shape=(64, 64, 3))

# Freeze most layers
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze last 2 layers
for layer in base_model.layers[len(base_model.layers) - 2:]:
    layer.trainable = True

print(f"\nUnfrozen layers: {sum([1 for layer in base_model.layers if layer.trainable])}")

# Add custom classification layers with dropout
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(102, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile with lower learning rate
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("\nModel Summary (Fine-tuning):")
model.summary()

# Train the model
print("\nTraining stage 2 (fine-tuning)...")
history2 = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    validation_data=(x_test, y_test)
)

# Evaluate stage 2
loss2, acc2 = model.evaluate(x_test, y_test)
print(f"\nStage 2 Results:")
print(f"  Loss: {loss2:.4f}")
print(f"  Accuracy: {acc2:.4f} ({acc2*100:.2f}%)")

# Plot training history comparison
print("\nGenerating training history plots...")
plt.figure(figsize=(16, 6))

# Stage 1 plots
plt.subplot(2, 2, 1)
plt.plot(history1.history['accuracy'], label='Train Accuracy')
plt.plot(history1.history['val_accuracy'], label='Val Accuracy')
plt.title('Stage 1: Model Accuracy (Frozen)', size=15)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(history1.history['loss'], label='Train Loss')
plt.plot(history1.history['val_loss'], label='Val Loss')
plt.title('Stage 1: Model Loss (Frozen)', size=15)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Stage 2 plots
plt.subplot(2, 2, 3)
plt.plot(history2.history['accuracy'], label='Train Accuracy')
plt.plot(history2.history['val_accuracy'], label='Val Accuracy')
plt.title('Stage 2: Model Accuracy (Fine-tuned)', size=15)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(history2.history['loss'], label='Train Loss')
plt.plot(history2.history['val_loss'], label='Val Loss')
plt.title('Stage 2: Model Loss (Fine-tuned)', size=15)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transfer_learning_training_comparison.png')
print("Training comparison saved as 'transfer_learning_training_comparison.png'")
plt.show()

# Make predictions
print("\n" + "=" * 70)
print("Making predictions...")
print("=" * 70)

predicted_value = model.predict(x_test)

# Display sample predictions
sample_indices = [60, 100, 150]
plt.figure(figsize=(15, 5))

for i, idx in enumerate(sample_indices):
    plt.subplot(1, 3, i + 1)
    plt.imshow(x_test[idx])
    predicted_class = labels[np.argmax(predicted_value[idx])]
    actual_class = labels[np.argmax(y_test[idx])]
    
    color = 'green' if predicted_class == actual_class else 'red'
    plt.title(f"Pred: {predicted_class}\nActual: {actual_class}", 
             size=10, color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('transfer_learning_predictions.png')
print("Sample predictions saved as 'transfer_learning_predictions.png'")
plt.show()

# Print detailed predictions
print("\nSample Predictions:")
print("-" * 70)
for idx in sample_indices:
    predicted_class = labels[np.argmax(predicted_value[idx])]
    actual_class = labels[np.argmax(y_test[idx])]
    confidence = np.max(predicted_value[idx]) * 100
    match = "✓" if predicted_class == actual_class else "✗"
    print(f"Sample {idx}: {match}")
    print(f"  Predicted: {predicted_class} (confidence: {confidence:.2f}%)")
    print(f"  Actual: {actual_class}")
    print()

# Final evaluation
print("=" * 70)
print("Final Results Summary")
print("=" * 70)
print(f"\nStage 1 (Frozen layers):")
print(f"  Accuracy: {acc1*100:.2f}%")
print(f"  Loss: {loss1:.4f}")

print(f"\nStage 2 (Fine-tuned):")
print(f"  Accuracy: {acc2*100:.2f}%")
print(f"  Loss: {loss2:.4f}")

print(f"\nImprovement: {(acc2-acc1)*100:.2f}%")

print("\n" + "=" * 70)
print("Assignment 6 (Transfer Learning) completed successfully!")
print("=" * 70)
