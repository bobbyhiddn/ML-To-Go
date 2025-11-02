# PyTorch vs TensorFlow Comparison

A practical comparison for document classification tasks.

## Overview

Both frameworks are excellent for ML development. This guide helps you choose based on your use case.

## Quick Comparison Table

| Feature | PyTorch | TensorFlow/Keras |
|---------|---------|------------------|
| **Learning Curve** | Moderate | Easier (Keras) |
| **Debugging** | Excellent (Pythonic) | Good (eager execution) |
| **Production** | Good (TorchServe) | Excellent (TF Serving) |
| **Research** | Dominant | Growing |
| **Community** | Large, research-focused | Large, industry-focused |
| **Mobile** | Good (PyTorch Mobile) | Excellent (TF Lite) |
| **Syntax** | More explicit | More abstracted |

## Code Comparison

### Loading a Pre-trained Model

**PyTorch:**
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
```

**TensorFlow:**
```python
from transformers import TFAutoModel, AutoTokenizer

model = TFAutoModel.from_pretrained('BAAI/bge-large-en-v1.5', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
```

### Building a Simple Classifier

**PyTorch:**
```python
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Classifier(768, 5)
```

**TensorFlow/Keras:**
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(768,)),
    keras.layers.Dense(5, activation='softmax')
])
```

### Training Loop

**PyTorch:**
```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
```

**TensorFlow/Keras:**
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset
)
```

## Strengths & Weaknesses

### PyTorch

**Strengths:**
- Intuitive, Pythonic syntax
- Excellent for research and experimentation
- Dynamic computation graphs (easier debugging)
- Strong NLP/Transformers ecosystem
- Popular in academia

**Weaknesses:**
- More boilerplate for training
- Production deployment requires extra tools
- Smaller mobile ecosystem

### TensorFlow/Keras

**Strengths:**
- High-level API (Keras) is beginner-friendly
- Excellent production tools (TF Serving, TF Lite)
- Strong industry adoption
- Better mobile/edge deployment
- TensorBoard for visualization

**Weaknesses:**
- Can feel less Pythonic
- Debugging can be harder (graph mode)
- More abstraction can hide details

## Recommendation for This Course

### Weeks 1-6: Use **PyTorch**
- Better for learning fundamentals
- Clearer view of what's happening
- Stronger sentence-transformers integration
- Easier experimentation

### Weeks 7-8: Either Framework
- SageMaker supports both equally
- Choose based on your preference from Weeks 2-3

## Framework Selection Guide

**Choose PyTorch if:**
- You value explicit control
- You're doing research or experiments
- You prefer Pythonic code
- NLP is your primary focus

**Choose TensorFlow if:**
- You're deploying to production soon
- You need mobile/edge deployment
- You prefer high-level abstractions
- You're in a TensorFlow-heavy organization

## Real-World Usage

**PyTorch dominates:**
- Academic research papers
- NLP research (BERT, GPT variants)
- Computer vision research

**TensorFlow dominates:**
- Production ML systems at scale
- Mobile applications
- Edge devices
- Established ML pipelines

## Migration Between Frameworks

Models trained in one framework can often be converted:

**PyTorch → TensorFlow:**
```python
# Using ONNX as intermediate format
import torch.onnx
import onnx
import onnx_tf

# Export PyTorch to ONNX
torch.onnx.export(pytorch_model, dummy_input, "model.onnx")

# Convert ONNX to TensorFlow
onnx_model = onnx.load("model.onnx")
tf_model = onnx_tf.backend.prepare(onnx_model)
```

**TensorFlow → PyTorch:**
```python
# Using ONNX or HuggingFace converters
# Less common, more complex
```

## Performance Comparison

For document classification tasks:
- **Training speed:** Comparable (both optimized)
- **Inference speed:** Comparable for CPU
- **Memory usage:** Similar
- **GPU utilization:** Both excellent

The performance difference is negligible for most use cases. Choose based on ecosystem and workflow preferences.

## Our Approach

This course teaches **both** frameworks:
- **Week 2:** PyTorch fundamentals
- **Week 3:** TensorFlow/Keras fundamentals
- **Weeks 4-8:** Your choice (examples provided for both)

By learning both, you'll be versatile and can choose the right tool for each job.

## Additional Resources

- **PyTorch:** pytorch.org/tutorials
- **TensorFlow:** tensorflow.org/tutorials
- **Comparison:** paperswithcode.com/trends
