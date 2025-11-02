# Week 2: PyTorch Fundamentals

## Overview

Learn to build neural networks with PyTorch. Progress from basic tensor operations to training your first text classifier.

## Learning Objectives

- Create and manipulate PyTorch tensors
- Build neural networks with nn.Module
- Implement training loops with backpropagation
- Fine-tune a pre-trained model
- Achieve >85% accuracy on sentiment classification

## Notebooks

### 1. `01_tensors_operations.ipynb`
- Creating tensors
- Mathematical operations
- GPU acceleration (optional)
- Autograd for automatic differentiation

### 2. `02_simple_neural_network.ipynb`
- Building networks with nn.Module
- Layers: Linear, ReLU, Dropout
- Forward pass implementation
- Model architecture design

### 3. `03_training_loop.ipynb`
- Loss functions
- Optimizers (SGD, Adam)
- Training and validation loops
- Saving/loading checkpoints

### 4. `04_text_classification.ipynb`
- Working with text data
- Using pre-trained embeddings
- Building a document classifier
- Evaluation metrics

## Exercises

### `build_sentiment_classifier.py`
Create a sentiment classifier:
- Binary classification (positive/negative)
- Use pre-trained embeddings (frozen)
- 2 hidden layers
- Achieve >85% test accuracy

### `train_document_classifier.py`
Train a multi-class classifier:
- 5+ categories
- Custom dataset
- Training/validation split
- Save best model

## Key Concepts

**Tensors:** Multi-dimensional arrays, like numpy but with GPU support

**nn.Module:** Base class for all neural network models

**Loss Function:** Measures how wrong predictions are

**Optimizer:** Updates model weights to minimize loss

**Backpropagation:** Automatic gradient computation for training

## Time Estimate
~16 hours

## Resources
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Start here:** `notebooks/01_tensors_operations.ipynb`
