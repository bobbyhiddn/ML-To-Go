# ML Document Classification Architecture

## System Overview

This document outlines the complete architecture for a production document classification system, from local experimentation to cloud deployment.

## Architecture Evolution

### Phase 1: Local Experimentation (Weeks 1-6)

```
┌─────────────────┐
│  Input Files    │
│  (PDF, DOCX,    │
│   XLSX, etc.)   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Text Extraction │
│ - PyPDF2        │
│ - python-docx   │
│ - openpyxl      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Text Embedding  │
│ - Sentence      │
│   Transformers  │
│ - BGE/E5 models │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Classification  │
│ - Cosine        │
│   Similarity    │
│ - Neural Net    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Output        │
│ - Labels        │
│ - Confidence    │
│ - Metadata      │
└─────────────────┘
```

### Phase 2: Production Deployment (Weeks 7-8)

```
                        ┌─────────────────┐
                        │  API Gateway    │
                        └────────┬────────┘
                                 │
                                 v
┌────────────────────────────────────────────────────┐
│                   AWS SageMaker                     │
│                                                      │
│  ┌──────────────┐    ┌──────────────┐              │
│  │  Endpoint 1  │    │  Endpoint 2  │              │
│  │  (Client A)  │    │  (Client B)  │   ...        │
│  └──────┬───────┘    └──────┬───────┘              │
│         │                   │                       │
│         v                   v                       │
│  ┌──────────────────────────────────┐              │
│  │    Shared Model Container        │              │
│  │  - Text Extraction               │              │
│  │  - Embedding Model               │              │
│  │  - Classification Logic          │              │
│  └──────────────────────────────────┘              │
│                                                      │
└────────────────────────────────────────────────────┘
                        │
                        v
            ┌──────────────────────┐
            │   S3 Storage         │
            │ - Model Artifacts    │
            │ - Batch Results      │
            │ - Logs               │
            └──────────────────────┘
```

## Component Details

### 1. Text Extraction Module

**Purpose:** Convert documents to plain text

**Components:**
- `pdf_extractor.py` - PDF text extraction
- `docx_extractor.py` - Word document extraction
- `xlsx_extractor.py` - Excel spreadsheet extraction
- `unified.py` - Unified interface for all formats

**Key Features:**
- Automatic format detection
- Fallback strategies (e.g., OCR for scanned PDFs)
- Metadata preservation
- Error handling

**Technology Stack:**
- PyPDF2, pdfplumber (PDFs)
- python-docx (Word)
- openpyxl (Excel)
- pytesseract (OCR)

### 2. Embedding Module

**Purpose:** Convert text to semantic vectors

**Components:**
- `embedder.py` - Embedding generation
- Model selection logic
- Batch processing
- Caching layer

**Key Features:**
- Support for multiple models (BGE, E5, MiniLM)
- Batch inference for efficiency
- Model quantization (Week 6)
- Semantic caching

**Technology Stack:**
- sentence-transformers
- PyTorch or TensorFlow
- HuggingFace transformers

### 3. Classification Module

**Purpose:** Assign categories to documents

**Components:**
- `classifier.py` - Classification logic
- Confidence scoring
- Multi-label support (optional)

**Approaches:**
1. **Zero-shot (Week 1):** Cosine similarity to category descriptions
2. **Fine-tuned (Weeks 2-3):** Trained neural network classifier
3. **Hybrid:** Combination of both

**Key Features:**
- Confidence thresholds
- Low-confidence flagging
- Batch classification
- Category management

### 4. Evaluation Module

**Purpose:** Measure and improve performance

**Components:**
- `evaluator.py` - Metrics calculation
- Confusion matrix generation
- Error analysis

**Metrics:**
- Accuracy
- Precision/Recall/F1 per category
- Confusion matrix
- Confidence calibration

### 5. Pipeline Orchestration

**Purpose:** End-to-end processing

**Flow:**
```python
for file in input_directory:
    text = extract_text(file)
    embedding = generate_embedding(text)
    category, confidence = classify(embedding)

    if confidence > HIGH_THRESHOLD:
        move_to_folder(file, category)
    elif confidence > LOW_THRESHOLD:
        flag_for_review(file, category, confidence)
    else:
        move_to_manual_review(file)
```

**Features:**
- Progress tracking
- Error recovery
- Parallel processing (Week 6)
- Logging and monitoring

## Data Flow

### Training Phase (Weeks 2-3)

```
Labeled Data → Text Extraction → Embeddings → Train/Val Split
                                                     │
                                                     v
                                              Training Loop
                                                     │
                                                     v
                                              Model Checkpoint
```

### Inference Phase

```
New Document → Text Extraction → Embedding → Classification → Output
                                                  │
                                                  v
                                            (Optional)
                                         Model Fine-tuning
```

## Deployment Architecture (AWS SageMaker)

### Components

1. **Docker Container**
   - Base image: python:3.11-slim
   - Includes: model, dependencies, inference script
   - Entry point: Flask/FastAPI server

2. **SageMaker Model**
   - Links to container in ECR
   - Points to model artifacts in S3

3. **SageMaker Endpoint**
   - Real-time inference
   - Auto-scaling configuration
   - Instance type: ml.m5.large (adjustable)

4. **Batch Transform Job** (Optional)
   - Process large batches
   - Cost-effective for bulk processing
   - Output to S3

### Multi-Tenant Design

```
API Request (Client ID: A)
    │
    v
Route to Endpoint A
    │
    v
Load Client A's category definitions
    │
    v
Classify with shared model
    │
    v
Return results
```

**Key Considerations:**
- Shared model container (cost efficiency)
- Client-specific category mappings in S3
- Isolated data storage per client
- Usage tracking and billing

## Scaling Considerations

### Local (Development)

- **Throughput:** 100-1000 docs/hour
- **Hardware:** 8GB RAM, CPU only
- **Cost:** $0 (local machine) or $0.18/hr (Codespace)

### SageMaker (Production)

- **Throughput:** 10,000+ docs/hour
- **Hardware:** ml.m5.large or larger
- **Cost:** ~$0.10/hour (ml.m5.large) + inference costs

### Optimization Strategies

1. **Model Quantization** - Reduce model size by 4x
2. **Batch Processing** - Process 32-64 docs at once
3. **Caching** - Cache embeddings for repeated docs
4. **Auto-scaling** - Scale endpoints based on load

## Monitoring & Observability

### Metrics to Track

1. **Performance:**
   - Inference latency (p50, p90, p99)
   - Throughput (requests/second)
   - Error rate

2. **Quality:**
   - Classification accuracy
   - Confidence distribution
   - Low-confidence rate

3. **Cost:**
   - Endpoint runtime hours
   - Inference request count
   - Data transfer (S3)

### Tools

- **Development:** Python logging, tqdm progress bars
- **Production:** CloudWatch, SageMaker Model Monitor

## Security Considerations

1. **Data Privacy:**
   - Encrypt data at rest (S3)
   - Encrypt data in transit (HTTPS)
   - Client data isolation

2. **Access Control:**
   - IAM roles for SageMaker
   - API keys for endpoints
   - VPC configuration (optional)

3. **Secrets Management:**
   - AWS Secrets Manager for credentials
   - No hardcoded keys in code

## Future Enhancements

1. **Active Learning:** Flag uncertain predictions for labeling
2. **Model Updates:** Continuous training pipeline
3. **Multi-modal:** Add image classification (scanned docs)
4. **Real-time Monitoring:** Dashboard for classification quality
5. **A/B Testing:** Compare model versions in production

## Technology Stack Summary

**Development (Weeks 1-6):**
- Python 3.11
- PyTorch / TensorFlow
- sentence-transformers
- Jupyter notebooks
- pytest

**Production (Weeks 7-8):**
- AWS SageMaker
- Docker
- Flask/FastAPI
- S3 for storage
- CloudWatch for monitoring

## Cost Estimates

**Development:**
- GitHub Codespaces: Free tier or $0.18/hour
- Model downloads: Free (HuggingFace)

**Production (per client):**
- SageMaker endpoint: ~$72/month (ml.m5.large, always on)
- Storage (S3): ~$0.023/GB/month
- Inference: ~$0.0004 per 1000 requests

**Recommended Starting Point:**
- Use Batch Transform (on-demand) instead of persistent endpoints
- Scale to real-time endpoints as usage grows

## Next Steps

1. Complete Weeks 1-6 to build local system
2. Week 7: Containerize your best model
3. Week 8: Deploy to SageMaker with test client
4. Production: Add monitoring, multi-tenancy, optimization
