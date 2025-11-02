# ML-To-Go: Document Classification Learning Path

A comprehensive 8-week learning path for mastering document classification using machine learning, from local development with PyTorch/TensorFlow to cloud deployment with AWS SageMaker.

## 🎯 Overview

This repository provides a structured approach to learning document classification, covering:
- Text embeddings and feature extraction
- Classical machine learning algorithms
- Deep learning with neural networks and transformers
- Cloud deployment with AWS SageMaker

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git
- (Optional) Docker for devcontainer support
- (For weeks 7-8) AWS account

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bobbyhiddn/ML-To-Go.git
cd ML-To-Go
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. (Optional) Use the devcontainer:
   - Open in VS Code with Remote-Containers extension
   - Select "Reopen in Container"

## 📚 8-Week Learning Path

### Weeks 1-6: Local Development (PyTorch/TensorFlow)

#### Week 1: Document Embeddings
**Focus**: Understanding text embeddings and vector representations

- Learn about word2vec, GloVe, and sentence embeddings
- Implement document embedding using sentence-transformers
- Explore similarity metrics and semantic search
- Visualize embeddings in 2D space

📁 Directory: `week1-embeddings/`

#### Week 2: Feature Extraction
**Focus**: Extracting meaningful features from documents

- Parse PDF and DOCX documents
- Implement TF-IDF and bag-of-words
- Learn n-gram features and feature engineering
- Build feature extraction pipelines

📁 Directory: `week2-feature-extraction/`

#### Week 3: Classical Machine Learning
**Focus**: Traditional ML algorithms for classification

- Train SVM, Random Forest, and Naive Bayes classifiers
- Implement cross-validation and hyperparameter tuning
- Evaluate models with appropriate metrics
- Compare algorithm performance

📁 Directory: `week3-classical-ml/`

#### Week 4: Neural Networks for Text
**Focus**: Deep learning fundamentals

- Build feedforward neural networks
- Implement CNNs and RNNs for text
- Understand training loops and optimizers
- Compare with classical ML approaches

📁 Directory: `week4-neural-networks/`

#### Week 5: Transformer Models
**Focus**: Modern NLP with transformers

- Understand transformer architecture
- Use pre-trained models (BERT, RoBERTa, DistilBERT)
- Implement document classification with transformers
- Analyze attention mechanisms

📁 Directory: `week5-transformers/`

#### Week 6: Fine-Tuning Transformers
**Focus**: Transfer learning and model optimization

- Fine-tune pre-trained models on custom data
- Implement learning rate schedules
- Track training metrics and visualize learning curves
- Save and deploy fine-tuned models

📁 Directory: `week6-fine-tuning/`

### Weeks 7-8: Cloud Deployment (AWS SageMaker)

#### Week 7: Introduction to AWS SageMaker
**Focus**: Cloud-based ML training

- Set up AWS SageMaker environment
- Prepare data for cloud training
- Run training jobs on SageMaker
- Monitor training with CloudWatch

📁 Directory: `week7-sagemaker-intro/`

#### Week 8: SageMaker Model Deployment
**Focus**: Production ML deployment

- Deploy models as SageMaker endpoints
- Implement real-time and batch inference
- Monitor deployed models
- Understand MLOps best practices

📁 Directory: `week8-sagemaker-deploy/`

## 📂 Repository Structure

```
ML-To-Go/
├── .devcontainer/          # Development container configuration
│   └── devcontainer.json   # Python 3.11 + Jupyter setup
├── week1-embeddings/       # Week 1: Document embeddings
├── week2-feature-extraction/  # Week 2: Feature engineering
├── week3-classical-ml/     # Week 3: Traditional ML algorithms
├── week4-neural-networks/  # Week 4: Deep learning basics
├── week5-transformers/     # Week 5: Transformer models
├── week6-fine-tuning/      # Week 6: Transfer learning
├── week7-sagemaker-intro/  # Week 7: AWS SageMaker setup
├── week8-sagemaker-deploy/ # Week 8: Model deployment
├── experiments/            # Experiment outputs and logs
├── sample-data/            # Training and test datasets
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore patterns
└── README.md              # This file
```

## 🛠️ Technologies

### Machine Learning Frameworks
- **PyTorch**: Deep learning framework
- **TensorFlow**: End-to-end ML platform
- **scikit-learn**: Classical ML algorithms

### NLP Libraries
- **sentence-transformers**: Sentence and document embeddings
- **transformers**: Pre-trained transformer models (Hugging Face)

### Document Processing
- **pypdf2**: PDF parsing and text extraction
- **python-docx**: Word document processing

### Cloud Services
- **AWS SageMaker**: Managed ML training and deployment
- **boto3**: AWS SDK for Python

### Development Tools
- **Jupyter**: Interactive notebooks
- **matplotlib/seaborn**: Data visualization

## 📊 Example Workflows

### Basic Document Classification
```python
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(documents)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels)
classifier = SVC()
classifier.fit(X_train, y_train)
```

### Transformer Fine-Tuning
```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16
)

# Train
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
```

## 🎓 Learning Resources

### General ML/NLP
- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### Transformers
- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

### AWS SageMaker
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Examples Repository](https://github.com/aws/amazon-sagemaker-examples)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new examples and tutorials
- Improve documentation
- Fix bugs or issues
- Share interesting datasets

## 📝 License

This project is open source and available for educational purposes.

## 🔗 Additional Resources

- **Datasets**: Kaggle, Hugging Face Datasets, Papers With Code
- **Communities**: r/MachineLearning, Hugging Face Forums, PyTorch Forums
- **Tools**: Weights & Biases, MLflow, TensorBoard

## 💡 Tips for Success

1. **Start Simple**: Begin with classical ML before diving into deep learning
2. **Experiment Often**: Use the experiments/ folder to track different approaches
3. **Document Your Work**: Keep notebooks organized and well-commented
4. **Use Version Control**: Commit regularly and use meaningful messages
5. **Monitor Resources**: Be mindful of compute costs, especially with SageMaker
6. **Join Communities**: Learn from others and share your progress

Happy Learning! 🚀 
