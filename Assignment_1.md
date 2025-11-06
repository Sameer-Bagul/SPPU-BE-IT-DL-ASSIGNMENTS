# Deep Learning Frameworks — Summary & Comparison

A concise overview of four popular deep learning frameworks, their distinct features, core functionality, and current status.

---

## 1. TensorFlow

**Distinct features**
- Open-source framework by Google with a comprehensive ecosystem.
- Supports distributed computing and deployment to mobile/embedded devices.
- Offers high-level APIs (tf.keras) and low-level APIs for flexibility.

**Functionality**
- Graph-based computation (historically); TensorFlow 2 uses eager execution with `tf.function` for graph performance.
- Extensive libraries for neural networks (layers, optimizers) and deployment tools.

---

## 2. Keras

**Distinct features**
- High-level neural networks API designed for rapid prototyping.
- Simple, consistent, and user-friendly interface.
- Runs on top of backends (now primarily TensorFlow via `tf.keras`).

**Functionality**
- Great for beginners and quick model development.
- Provides concise APIs to define, train, and evaluate models with minimal code.

---

## 3. Theano

**Distinct features**
- Early open-source numerical computation library focused on optimizing mathematical expressions.
- Specialized in symbolic math and optimizing CPU/GPU computations.

**Functionality**
- Efficient for symbolic mathematics; used historically as a backend for frameworks like Keras.
- Largely unmaintained today; recommended only for legacy projects or research forks.

---

## 4. PyTorch

**Distinct features**
- Open-source framework from Meta (Facebook AI Research).
- Dynamic computation graph (eager by default) enabling flexible model construction and debugging.

**Functionality**
- Ideal for research and rapid experimentation.
- Can be scripted (TorchScript) for production deployment; strong ecosystem (torchvision, torchtext, etc.).

---

## Comparison Table

| Framework   | Distinct features (short)                                         | Computation graph / execution                              | Typical use cases                       | Current status / backend                        |
|-------------|-------------------------------------------------------------------|------------------------------------------------------------|-----------------------------------------|-------------------------------------------------|
| TensorFlow  | Comprehensive ecosystem, distributed support, mobile/embedded     | Graph-based historically; TF2 supports eager + `tf.function` | Production, large-scale/deployed ML     | Actively developed by Google; primary backend for `tf.keras` |
| Keras       | High-level, simple API for rapid prototyping                      | Uses backend engine (now primarily TensorFlow via `tf.keras`) | Beginner-friendly model building, quick prototyping | Integrated into TensorFlow as `tf.keras`        |
| Theano      | Symbolic math optimizer, early deep learning backend              | Static symbolic graph                                      | Legacy research, historical backend     | Largely unmaintained (community forks exist)    |
| PyTorch     | Dynamic graph, easy debugging, strong research focus              | Eager by default; can be scripted (TorchScript)            | Research, rapid experimentation, growing production use | Actively developed by Meta; rich ecosystem      |

---

References: framework names and typical characteristics are summarized from common public documentation and community usage.
