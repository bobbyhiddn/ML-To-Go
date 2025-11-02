# ML Document Classification Training

A comprehensive 8-week training program to build production-grade semantic document classification systems. Start with local experimentation using PyTorch and TensorFlow, then graduate to AWS SageMaker for scalable deployment.

## 🎯 Learning Objectives

**Weeks 1-6: Local Development & Fundamentals**
- Master embedding models and semantic similarity
- Build classifiers in PyTorch and TensorFlow
- Extract text from any document format
- Implement complete classification pipelines
- Optimize models for speed and efficiency

**Weeks 7-8: Cloud Deployment**
- Package models for SageMaker
- Deploy scalable endpoints
- Build multi-tenant architectures
- Monitor production systems

## 🚀 Quick Start

### Prerequisites
- GitHub account with Codespaces access
- Basic Python knowledge
- (Optional) AWS account for weeks 7-8

### Setup

1. **Start Codespace**
   - Click "Code" → "Codespaces" → "Create codespace"
   - Wait for automatic setup (installs all dependencies)

2. **Verify Installation**
   ```bash
   python --version  # Should be 3.11+
   jupyter --version
   python -c "import torch; print(torch.__version__)"
   python -c "import tensorflow; print(tensorflow.__version__)"
   ```

3. **Start Jupyter**
   ```bash
   jupyter notebook
   ```
   Access at http://localhost:8888

4. **Begin Week 1**
   Open `week1-embeddings/notebooks/01_what_are_embeddings.ipynb`

## 📚 Weekly Curriculum

### Week 1: Embedding Models
**Goal:** Understand how text becomes vectors and semantic similarity

**Topics:**
- What are embeddings?
- Sentence transformers
- Model comparison (BGE, E5, MiniLM)
- Cosine similarity for classification

**Deliverable:** Compare 5 embedding models on 100 documents

### Week 2: PyTorch Basics
**Goal:** Build neural networks with PyTorch

**Topics:**
- Tensors and operations
- Building networks with nn.Module
- Training loops
- Text classification from scratch

**Deliverable:** Sentiment classifier achieving >85% accuracy

### Week 3: TensorFlow Basics
**Goal:** Build neural networks with TensorFlow/Keras

**Topics:**
- TensorFlow fundamentals
- Keras Sequential and Functional APIs
- Text classification with Keras
- PyTorch vs TensorFlow comparison

**Deliverable:** Same classifier in TensorFlow, compare frameworks

### Week 4: Text Extraction Pipeline
**Goal:** Extract text from any document type

**Topics:**
- PDF extraction (PyPDF2, pdfplumber)
- Office documents (Word, Excel, PowerPoint)
- OCR for scanned documents
- Unified extraction interface

**Deliverable:** Process 1000+ mixed documents

### Week 5: Classification Engine
**Goal:** Build end-to-end classification system

**Topics:**
- Complete pipeline (extract → embed → classify)
- Confidence thresholds
- Evaluation metrics
- Error analysis

**Deliverable:** Classify 1000 files into 10 folders with >80% accuracy

### Week 6: Optimization
**Goal:** Make models faster and more efficient

**Topics:**
- Model quantization
- Batch processing strategies
- Caching and deduplication
- Performance benchmarking

**Deliverable:** 2x speed improvement from Week 5

### Week 7: SageMaker Preparation
**Goal:** Package models for cloud deployment

**Topics:**
- AWS setup and IAM roles
- Containerizing models with Docker
- SageMaker inference scripts
- Local container testing

**Deliverable:** Working Docker container for your classifier

### Week 8: SageMaker Deployment
**Goal:** Deploy production-ready system

**Topics:**
- SageMaker endpoints
- Batch Transform jobs
- Multi-tenant architecture
- Monitoring and cost optimization

**Deliverable:** Production classification service on AWS

## 🧪 Experiments Folder

Use `experiments/` for quick tests with Claude Code:

```bash
cd experiments
claude-code

"Create a script to compare BGE-large vs E5-large on 50 documents"
```

See `docs/claude-code-prompts.md` for more examples.

## 📱 Mobile Development

This repository works great with GitHub Codespaces on mobile:

- **Quick reviews:** GitHub Mobile app
- **Light editing:** Mobile browser with Codespaces
- **Full development:** Tablet or laptop

## 🔬 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_extractors.py
```

## 📊 Progress Tracking

- [ ] Week 1: Embeddings mastered
- [ ] Week 2: PyTorch classifier built
- [ ] Week 3: TensorFlow classifier built
- [ ] Week 4: Text extraction working
- [ ] Week 5: Complete pipeline operational
- [ ] Week 6: Optimizations implemented
- [ ] Week 7: Model containerized
- [ ] Week 8: Deployed to SageMaker

## 🎓 Learning Resources

- **Embedding Models:** sentence-transformers documentation
- **PyTorch:** pytorch.org/tutorials
- **TensorFlow:** tensorflow.org/tutorials
- **SageMaker:** AWS SageMaker documentation

## 💡 Tips for Success

1. **Start with notebooks** - Understand concepts interactively
2. **Use Claude Code** - Get unstuck quickly on syntax
3. **Build incrementally** - Each week builds on the last
4. **Test frequently** - Don't wait until the end to validate
5. **Document learnings** - Keep notes on what works

## 🤝 Getting Help

- Check `docs/` folder for guides
- Review completed notebooks for examples
- Use Claude Code for coding questions
- Test in `experiments/` before committing

## 📈 Business Context

This training prepares you to build a **multi-tenant document classification service** that:
- Ingests terabytes of unorganized data
- Automatically classifies by semantic meaning
- Scales to dozens of clients
- Delivers organized data for downstream products

**Target market:** Government contractors, enterprises with data chaos

## 🔐 Security Notes

- Never commit AWS credentials
- Use `.env` for sensitive data
- Keep `.env.example` updated
- Review `.gitignore` before committing

## 📝 License

MIT License - See LICENSE file

---

**Ready to start? Open Week 1 →** `week1-embeddings/README.md` 
