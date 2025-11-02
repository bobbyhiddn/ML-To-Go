# Claude Code Prompts for ML Classification

Effective prompts for using Claude Code throughout your training.

## Week 1: Embeddings

### Compare Embedding Models
```
Create a Python script that:
1. Loads three embedding models: all-MiniLM-L6-v2, all-mpnet-base-v2, and BAAI/bge-large-en-v1.5
2. Embeds a list of 20 sample documents
3. Measures embedding time for each model
4. Compares memory usage
5. Tests classification accuracy on 100 pre-labeled documents
6. Outputs a comparison table with recommendations
```

### Build First Classifier
```
Help me build a zero-shot document classifier:
1. Load sentence-transformers model
2. Define 5 folder categories with descriptions
3. Embed the folder descriptions
4. For each input document, find most similar folder
5. Return classification with confidence score
6. Handle edge cases (low confidence, ties)
```

## Week 2: PyTorch

### Simple Neural Network
```
Create a PyTorch text classifier that:
1. Uses a pre-trained embedding layer (frozen)
2. Adds a feedforward network (2 hidden layers)
3. Outputs classification for 5 categories
4. Includes training loop with validation
5. Shows training/validation loss curves
6. Saves best model checkpoint
```

### Training Loop
```
Build a complete PyTorch training loop with:
1. DataLoader for text data
2. Loss function (CrossEntropyLoss)
3. Optimizer (Adam with learning rate schedule)
4. Training and validation steps
5. Early stopping logic
6. Progress bars with tqdm
7. Model checkpointing
```

## Week 3: TensorFlow

### Keras Classifier
```
Rebuild my PyTorch classifier in TensorFlow/Keras:
1. Same architecture (embedding + 2 dense layers)
2. Use functional API
3. Compile with appropriate loss and metrics
4. Train with callbacks (EarlyStopping, ModelCheckpoint)
5. Plot training history
6. Compare results to PyTorch version
```

## Week 4: Text Extraction

### PDF Extractor
```
Create a robust PDF text extractor that:
1. Tries PyPDF2 first, falls back to pdfplumber
2. Detects if PDF is scanned (needs OCR)
3. Extracts metadata (title, author, dates)
4. Handles encrypted PDFs gracefully
5. Returns structured dict with text and metadata
6. Includes comprehensive error handling
```

### Unified Extractor
```
Build a unified document extractor class:
1. Auto-detects file type from extension
2. Routes to appropriate extractor (PDF, DOCX, XLSX, etc)
3. Returns standardized format: {text, metadata, file_type}
4. Logs extraction stats (time, characters extracted)
5. Handles corrupt/unsupported files
6. Includes unit tests for each file type
```

## Week 5: Classification Pipeline

### End-to-End Pipeline
```
Create a complete classification pipeline:
1. Takes input directory of mixed files
2. Extracts text from each file
3. Generates embeddings
4. Classifies to folder structure
5. Applies confidence thresholds
6. Generates classification report (CSV)
7. Flags low-confidence files for review
8. Includes progress bars and logging
```

### Evaluation System
```
Build a classification evaluator that:
1. Takes true labels and predicted labels
2. Calculates accuracy, precision, recall, F1
3. Generates confusion matrix
4. Identifies most common errors
5. Shows per-class performance
6. Suggests improvements based on errors
7. Visualizes results with seaborn
```

## Week 6: Optimization

### Model Quantization
```
Help me quantize my embedding model:
1. Load the full precision model
2. Apply dynamic quantization
3. Compare model sizes (before/after)
4. Benchmark inference speed
5. Test accuracy degradation
6. Show memory usage comparison
7. Recommend best quantization approach
```

### Batch Processing
```
Optimize my pipeline for batch processing:
1. Process files in batches of 32
2. Use multiprocessing for extraction
3. Batch embeddings efficiently
4. Add progress tracking
5. Handle failures gracefully (don't stop entire batch)
6. Benchmark vs sequential processing
7. Show speedup metrics
```

## Week 7: Docker & SageMaker Prep

### Dockerfile Creation
```
Create a Dockerfile for my classification model:
1. Base image: python:3.11-slim
2. Install dependencies from requirements.txt
3. Copy model files and inference script
4. Expose port 8080
5. Set up ENTRYPOINT for SageMaker
6. Optimize image size (multi-stage build)
7. Include healthcheck endpoint
```

### SageMaker Inference Script
```
Write a SageMaker inference script (serve.py):
1. Loads embedding model on startup
2. Handles /ping for health checks
3. Handles /invocations for inference
4. Accepts JSON input with text field
5. Returns classification and confidence
6. Includes error handling
7. Logs all requests for monitoring
```

## Week 8: Production Deployment

### Endpoint Deployment
```
Create a SageMaker endpoint deployment script:
1. Upload model artifacts to S3
2. Create SageMaker model from container
3. Create endpoint configuration (ml.m5.large)
4. Deploy endpoint
5. Test with sample data
6. Set up auto-scaling policy
7. Show how to invoke endpoint from code
```

### Monitoring Dashboard
```
Build a monitoring dashboard for my SageMaker endpoint:
1. Track invocation count
2. Monitor latency (p50, p90, p99)
3. Show error rates
4. Calculate cost per 1000 invocations
5. Alert on anomalies
6. Use CloudWatch metrics
7. Visualize with matplotlib or Streamlit
```

## General Tips for Claude Code

### Effective Prompts
- Be specific about inputs and outputs
- Mention error handling needs
- Request logging/progress bars
- Ask for tests when appropriate
- Specify visualization preferences

### When to Use Claude Code
- Boilerplate code (data loaders, training loops)
- Unfamiliar libraries (new API syntax)
- Debugging (ask Claude to review your code)
- Optimization (ask for performance improvements)

### When to Write Yourself
- Core learning concepts (your first classifier)
- Understanding fundamentals (how embeddings work)
- Architecture decisions (design choices)
- Experimentation (trying different approaches)

Remember: Claude Code is a tool to accelerate, not replace, learning.
