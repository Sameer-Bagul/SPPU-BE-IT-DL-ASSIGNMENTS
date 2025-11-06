# SPPU BE IT Deep Learning Assignments 🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**BE IT (2019 Course) || 414447: Lab Practice IV**

Complete repository of Deep Learning assignments for Savitribai Phule Pune University (SPPU) BE IT students. This repository contains implementations of various neural network architectures, from basic feedforward networks to advanced transfer learning models.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Setup Instructions](#setup-instructions)
- [Assignments Overview](#assignments-overview)
- [Dataset Information](#dataset-information)
- [Running the Notebooks](#running-the-notebooks)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** (Recommended: Python 3.9 or 3.10)
- **pip** (Python package manager)
- **Jupyter Notebook** or **VS Code** with Jupyter extension
- **Git** (for cloning the repository)
- **8GB+ RAM** (recommended for deep learning tasks)
- **CUDA-compatible GPU** (optional, but recommended for faster training)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/SPPU-BE-IT-DL-ASSIGNMENTS.git
cd SPPU-BE-IT-DL-ASSIGNMENTS

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 4. Install required packages
pip install -r requirements.txt

# 5. Launch Jupyter Notebook or open in VS Code
jupyter notebook
# OR open the folder in VS Code
```

---

## ⚙️ Setup Instructions

### Step 1: Environment Setup

**Option A: Using Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Verify activation (you should see (venv) in your terminal)
```

**Option B: Using Conda**
```bash
# Create conda environment
conda create -n dl-assignments python=3.9

# Activate it
conda activate dl-assignments
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

**Required Packages:**
- `tensorflow` - Deep learning framework
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical computations
- `pillow` - Image processing
- `scikit-learn` - Machine learning utilities
- `seaborn` - Statistical data visualization

### Step 3: Dataset Setup

The datasets are already organized in the `data/` folder:
- **MNIST**: Available through TensorFlow/Keras datasets (auto-downloaded)
- **CIFAR-10**: Available through TensorFlow/Keras datasets (auto-downloaded)
- **Caltech-101**: Located in `data/caltech-101-img/`
- **Credit Card Fraud**: Located in `data/creditcardfraud-csv/`
- **ECG Data**: Located in `data/ecg-csv/`
- **VGG16 Weights**: Located in `data/vgg/`

---

## 📚 Assignments Overview

### Assignment 1: Study of Deep Learning Packages 📚
**File:** `Assignment_1.md`

Comprehensive documentation comparing deep learning frameworks:
- **TensorFlow** - Google's production-ready framework
- **Keras** - High-level API for rapid prototyping
- **Theano** - Historical symbolic computation library
- **PyTorch** - Research-friendly dynamic framework

**What you'll learn:**
- Feature comparison between frameworks
- Use cases for each framework
- Current industry standards

---

### Assignment 2 & 3: Feedforward Neural Networks 🧠
**Files:** 
- `Assignment_2_3_MNIST.ipynb` - MNIST digit classification
- `Assignment_2_3_CIFAR10.ipynb` - CIFAR-10 image classification
- `Assignment_2_3_MNIST_offline_dataset.ipynb` - MNIST with local data
- `Assignment_2_3_CIFAR10_offline_dataset.ipynb` - CIFAR-10 with local data

**Objectives:**
- Build feedforward neural networks using Keras/TensorFlow
- Load and preprocess MNIST and CIFAR-10 datasets
- Define network architecture with Dense layers
- Train models using SGD optimizer
- Evaluate model performance
- Visualize training metrics (loss & accuracy)

**Key Concepts:**
- Neural network layers
- Activation functions
- Backpropagation
- Gradient descent optimization
- Model evaluation metrics

**To Run:**
```bash
# Open in Jupyter or VS Code and run cells sequentially
# Choose between online (auto-download) or offline (local data) versions
```

---

### Assignment 4: Anomaly Detection with Autoencoders 🕵️
**Files:**
- `Assignment_4_ECG.ipynb` - ECG signal anomaly detection
- `Assignment_4_ECG_offline_dataset.ipynb` - ECG with local data
- `Assignment_4_CreditCard_offline_dataset.ipynb` - Credit card fraud detection

**Objectives:**
- Implement autoencoder architecture
- Detect anomalies in time-series data (ECG)
- Identify fraudulent credit card transactions
- Understand encoder-decoder paradigm

**Key Concepts:**
- Autoencoder architecture
- Latent space representation
- Reconstruction error
- Anomaly score calculation
- Threshold-based detection

**Datasets:**
- ECG signals for heartbeat anomaly detection
- Credit card transactions for fraud detection

---

### Assignment 5: Continuous Bag of Words (CBOW) 📝
**File:** `Assignment_5.ipynb`

**Objectives:**
- Implement CBOW model for word embeddings
- Prepare text data for NLP tasks
- Generate training data from context
- Train word2vec style embeddings

**Key Concepts:**
- Word embeddings
- Context windows
- Skip-gram vs CBOW
- Vector representations of words

---

### Assignment 6: Transfer Learning with CNNs 📷
**Files:**
- `Assignment_6_classification.ipynb` - Caltech-101 classification
- `Assignment_6_MNIST.ipynb` - MNIST with transfer learning

**Objectives:**
- Use pre-trained CNN models (VGG16, ResNet, etc.)
- Apply transfer learning to new datasets
- Freeze and fine-tune layers
- Add custom classification layers
- Optimize hyperparameters

**Key Concepts:**
- Transfer learning
- Feature extraction
- Fine-tuning
- Pre-trained models
- Domain adaptation

**Datasets:**
- Caltech-101: 101 object categories
- CIFAR-10: 10 classes of objects

**To Run:**
```bash
# Make sure VGG16 weights are in data/vgg/ folder
# Open notebook and run cells sequentially
```

---

## 📊 Dataset Information

### MNIST Dataset
- **Type**: Handwritten digits (0-9)
- **Images**: 70,000 (60k train, 10k test)
- **Size**: 28×28 grayscale
- **Download**: Automatic via TensorFlow/Keras

### CIFAR-10 Dataset
- **Type**: 10 object classes
- **Images**: 60,000 (50k train, 10k test)
- **Size**: 32×32 color
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Download**: Automatic via TensorFlow/Keras

### Caltech-101 Dataset
- **Type**: Object recognition
- **Categories**: 101 object classes
- **Location**: `data/caltech-101-img/`
- **Format**: JPG images (various sizes)

### ECG Dataset
- **Type**: Time-series ECG signals
- **Location**: `data/ecg-csv/ecg.csv`
- **Purpose**: Anomaly detection in heartbeat patterns

### Credit Card Fraud Dataset
- **Type**: Transaction data
- **Location**: `data/creditcardfraud-csv/creditcard.csv`
- **Purpose**: Fraud detection using autoencoders

---

## 🎮 Running the Notebooks

### Using Jupyter Notebook

```bash
# 1. Activate your virtual environment
source venv/bin/activate  # or conda activate dl-assignments

# 2. Launch Jupyter Notebook
jupyter notebook

# 3. Navigate to the desired notebook in your browser
# 4. Run cells sequentially using Shift+Enter
```

### Using VS Code

```bash
# 1. Open the project folder in VS Code
code .

# 2. Install Python and Jupyter extensions (if not already installed)
# 3. Open any .ipynb file
# 4. Select your Python interpreter (venv or conda environment)
# 5. Click "Run All" or run cells individually
```

### Execution Order

For each notebook:
1. **Import cells first** - Run all import statements
2. **Load data** - Execute data loading cells
3. **Preprocessing** - Run data preprocessing cells
4. **Model definition** - Define the network architecture
5. **Training** - Train the model (this may take time)
6. **Evaluation** - Evaluate and visualize results

### Expected Runtime

| Assignment | Approximate Time (CPU) | With GPU |
|------------|----------------------|----------|
| Assignment 2/3 (MNIST) | 5-10 minutes | 1-2 minutes |
| Assignment 2/3 (CIFAR-10) | 15-30 minutes | 3-5 minutes |
| Assignment 4 (Autoencoders) | 10-20 minutes | 2-4 minutes |
| Assignment 5 (CBOW) | 5-10 minutes | 1-2 minutes |
| Assignment 6 (Transfer Learning) | 20-40 minutes | 5-10 minutes |

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: No module named 'tensorflow'
# Solution:
pip install tensorflow

# Error: No module named 'sklearn'
# Solution:
pip install scikit-learn
```

#### 2. CUDA/GPU Issues
```bash
# Check if TensorFlow detects GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If no GPU detected but you have one, install GPU version:
pip install tensorflow[and-cuda]
```

#### 3. Memory Errors
```python
# Reduce batch size in your notebook
batch_size = 32  # Try reducing to 16 or 8

# Or enable memory growth for GPU
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### 4. Kernel Issues in Jupyter
```bash
# Install ipykernel in your virtual environment
pip install ipykernel

# Add your environment as a Jupyter kernel
python -m ipykernel install --user --name=venv --display-name "Python (DL Assignments)"
```

#### 5. Dataset Loading Issues
```python
# If automatic download fails, check internet connection
# For offline versions, ensure datasets are in correct folders:
# data/mnist-jpg/, data/cifar-10-img/, etc.
```

#### 6. Slow Training
- **Solution 1**: Reduce number of epochs
- **Solution 2**: Reduce model complexity (fewer layers/neurons)
- **Solution 3**: Use smaller batch size
- **Solution 4**: Use GPU if available

---

## 📁 Project Structure

```
SPPU-BE-IT-DL-ASSIGNMENTS/
│
├── Assignment_1.md                              # Framework comparison document
├── Assignment_2_3_MNIST.ipynb                   # MNIST classification
├── Assignment_2_3_MNIST_offline_dataset.ipynb   # MNIST with local data
├── Assignment_2_3_CIFAR10.ipynb                 # CIFAR-10 classification
├── Assignment_2_3_CIFAR10_offline_dataset.ipynb # CIFAR-10 with local data
├── Assignment_4_ECG.ipynb                       # ECG anomaly detection
├── Assignment_4_ECG_offline_dataset.ipynb       # ECG with local data
├── Assignment_4_CreditCard_offline_dataset.ipynb# Fraud detection
├── Assignment_5.ipynb                           # CBOW word embeddings
├── Assignment_6_classification.ipynb            # Transfer learning (Caltech)
├── Assignment_6_MNIST.ipynb                     # Transfer learning (MNIST)
├── requirements.txt                             # Python dependencies
├── README.md                                    # This file
│
└── data/                                        # Datasets directory
    ├── mnist-jpg/                              # MNIST images
    ├── cifar-10-img/                           # CIFAR-10 images
    ├── caltech-101-img/                        # Caltech-101 images
    ├── creditcardfraud-csv/                    # Credit card data
    ├── ecg-csv/                                # ECG time-series data
    └── vgg/                                    # Pre-trained VGG16 weights
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this repository:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

**Areas for contribution:**
- Bug fixes
- Additional datasets
- Performance optimizations
- Documentation improvements
- New assignment implementations

---

## 📖 Learning Resources

### Recommended Reading
- [Deep Learning Book by Ian Goodfellow](https://www.deeplearningbook.org/)
- [TensorFlow Official Documentation](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/guides/)

### Online Courses
- [Deep Learning Specialization - Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)
- [TensorFlow Developer Certificate](https://www.tensorflow.org/certificate)

### Video Tutorials
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Sentdex - Deep Learning with Python](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Feel free to use the code and materials for:
- Educational purposes
- Personal projects
- Academic assignments
- Research work

---

## 📧 Contact & Support

If you have any questions, suggestions, or need assistance:

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: [your-email@example.com]

---

## 🌟 Acknowledgments

- **SPPU BE IT Curriculum** - For providing comprehensive deep learning syllabus
- **TensorFlow & Keras Teams** - For excellent documentation and tools
- **Dataset Providers** - MNIST, CIFAR-10, Caltech-101, and other dataset creators
- **Open Source Community** - For continuous support and contributions

---

## 📈 Stats

![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/SPPU-BE-IT-DL-ASSIGNMENTS)
![GitHub issues](https://img.shields.io/github/issues/yourusername/SPPU-BE-IT-DL-ASSIGNMENTS)
![GitHub stars](https://img.shields.io/github/stars/yourusername/SPPU-BE-IT-DL-ASSIGNMENTS)

---

<p align="center">
  <b>Happy Learning! 🎓</b><br>
  Made with ❤️ for SPPU BE IT Students<br>
  <sub>© 2025 - All Rights Reserved</sub>
</p>

---

## 🔄 Version History

- **v1.0.0** (2025) - Initial release with all 6 assignments
- Added comprehensive documentation and setup instructions
- Included troubleshooting guide
- Organized dataset structure

---

**⭐ If you find this repository helpful, please consider giving it a star!**
