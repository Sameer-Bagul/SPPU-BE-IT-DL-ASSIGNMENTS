"""
Assignment 5: Continuous Bag of Words (CBOW) Model
Implementing word embeddings using CBOW approach
Dataset: Deep learning text corpus
"""

import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Data preparation
print("=" * 70)
print("Assignment 5: Continuous Bag of Words (CBOW) Model")
print("=" * 70)

data = """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional neural networks and Transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance."""

print("\nOriginal Text:")
print("-" * 70)
print(data)
print()

# Split into sentences
sentences = data.split('.')
print(f"Number of sentences: {len(sentences)}")

# Clean sentences
print("\nCleaning sentences...")
clean_sent = []
for sentence in sentences:
    if sentence == "":
        continue
    sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence)
    sentence = re.sub(r'(?:^| )\w (?:$| )', ' ', sentence).strip()
    sentence = sentence.lower()
    clean_sent.append(sentence)

print(f"Number of cleaned sentences: {len(clean_sent)}")
print("\nSample cleaned sentences:")
for i, sent in enumerate(clean_sent[:3]):
    print(f"{i+1}. {sent}")

# Tokenization
print("\n" + "=" * 70)
print("Tokenizing text...")
print("=" * 70)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_sent)
sequences = tokenizer.texts_to_sequences(clean_sent)

print(f"\nVocabulary size: {len(tokenizer.word_index)}")
print(f"Number of sequences: {len(sequences)}")

# Create word-index mappings
index_to_word = {}
word_to_index = {}

for i, sequence in enumerate(sequences):
    word_in_sentence = clean_sent[i].split()
    for j, value in enumerate(sequence):
        index_to_word[value] = word_in_sentence[j]
        word_to_index[word_in_sentence[j]] = value

print(f"\nTotal unique words: {len(index_to_word)}")
print("\nSample word-to-index mappings:")
sample_words = list(word_to_index.items())[:10]
for word, idx in sample_words:
    print(f"  '{word}' -> {idx}")

# Generate training data
print("\n" + "=" * 70)
print("Generating training data...")
print("=" * 70)

vocab_size = len(tokenizer.word_index) + 1
emb_size = 10
context_size = 2

contexts = []
targets = []

for sequence in sequences:
    for i in range(context_size, len(sequence) - context_size):
        target = sequence[i]
        context = [sequence[i - 2], sequence[i - 1], sequence[i + 1], sequence[i + 2]]
        contexts.append(context)
        targets.append(target)

print(f"Total training examples: {len(contexts)}")
print("\nSample context-target pairs:")
for i in range(5):
    words = []
    target = index_to_word.get(targets[i])
    for j in contexts[i]:
        words.append(index_to_word.get(j))
    print(f"  {words} -> '{target}'")

# Convert to numpy arrays
X = np.array(contexts)
Y = np.array(targets)

print(f"\nX shape: {X.shape}")
print(f"Y shape: {Y.shape}")

# Build CBOW model
print("\n" + "=" * 70)
print("Building CBOW model...")
print("=" * 70)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=emb_size, input_length=2*context_size),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

# Train the model
print("\n" + "=" * 70)
print("Training the model...")
print("=" * 70)

history = model.fit(X, Y, epochs=80, verbose=1)

# Plot training history
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], linewidth=2)
plt.title('Model Loss', size=20)
plt.xlabel('Epoch', size=15)
plt.ylabel('Loss', size=15)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], linewidth=2, color='green')
plt.title('Model Accuracy', size=20)
plt.xlabel('Epoch', size=15)
plt.ylabel('Accuracy', size=15)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cbow_training_history.png')
print("\nTraining history saved as 'cbow_training_history.png'")
plt.show()

# Visualize embeddings using PCA
print("\n" + "=" * 70)
print("Visualizing word embeddings...")
print("=" * 70)

embeddings = model.get_weights()[0]
print(f"Embeddings shape: {embeddings.shape}")

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 10))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)

# Annotate some words
sample_indices = list(range(1, min(20, len(index_to_word) + 1)))
for idx in sample_indices:
    if idx in index_to_word:
        word = index_to_word[idx]
        plt.annotate(word, (reduced_embeddings[idx, 0], reduced_embeddings[idx, 1]),
                    fontsize=10, alpha=0.7)

plt.title('Word Embeddings Visualization (PCA)', size=20)
plt.xlabel('First Principal Component', size=15)
plt.ylabel('Second Principal Component', size=15)
plt.grid(True, alpha=0.3)
plt.savefig('cbow_embeddings_visualization.png')
print("Embeddings visualization saved as 'cbow_embeddings_visualization.png'")
plt.show()

# Test the model
print("\n" + "=" * 70)
print("Testing the model...")
print("=" * 70)

test_sentences = [
    "known as structured learning",
    "transformers have applied to",
    "where they produced results",
    "cases surpassing expert performance"
]

print("\nTest predictions:")
print("-" * 70)

for sent in test_sentences:
    test_words = sent.split(" ")
    x_test = []
    for word in test_words:
        if word in word_to_index:
            x_test.append(word_to_index[word])
    
    if len(x_test) == 4:  # Need 4 context words
        x_test = np.array([x_test])
        pred = model.predict(x_test, verbose=0)
        pred_idx = np.argmax(pred[0])
        pred_word = index_to_word.get(pred_idx, "UNKNOWN")
        print(f"Context: {test_words}")
        print(f"Predicted word: '{pred_word}'")
        print()

print("=" * 70)
print("Assignment 5 (CBOW) completed successfully!")
print("=" * 70)
print(f"\nFinal Statistics:")
print(f"  Vocabulary size: {vocab_size}")
print(f"  Embedding dimension: {emb_size}")
print(f"  Training examples: {len(X)}")
print(f"  Final loss: {history.history['loss'][-1]:.4f}")
print(f"  Final accuracy: {history.history['accuracy'][-1]:.4f}")
