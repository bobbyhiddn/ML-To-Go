# Week 1: Understanding Embeddings

## Overview

Learn how text becomes numbers that machines can understand. Master embedding models and build your first zero-shot classifier using semantic similarity.

## Learning Objectives

By the end of this week, you will:
- Understand what embeddings are and why they matter
- Use sentence-transformers to embed text
- Compare different embedding models
- Build a zero-shot classifier using cosine similarity
- Achieve >70% accuracy without training a model

## Prerequisites

- Basic Python knowledge
- Understanding of vectors (high school math level)
- Jupyter notebook setup complete

## Notebooks

Work through these notebooks in order:

### 1. `01_what_are_embeddings.ipynb`
- What are embeddings?
- Visualization of word and sentence embeddings
- Distance metrics (cosine similarity, euclidean)
- Hands-on: Embed your first sentences

### 2. `02_sentence_transformers.ipynb`
- Introduction to sentence-transformers library
- Loading pre-trained models
- Generating embeddings for documents
- Understanding embedding dimensions

### 3. `03_model_comparison.ipynb`
- Compare 5+ embedding models
- Benchmarking speed and memory
- Quality assessment
- Choosing the right model for your use case

### 4. `04_cosine_similarity.ipynb`
- Deep dive into cosine similarity
- Building a zero-shot classifier
- Handling edge cases
- Confidence scoring

## Exercises

Complete these exercises to reinforce learning:

### `classify_10_documents.py`
Build a script that:
- Loads 10 sample documents
- Defines 5 category descriptions
- Classifies each document to a category
- Reports confidence scores

**Expected Output:**
```
Document: "Annual financial report..."
Category: Financial Reports
Confidence: 0.87
```

### `compare_three_models.py`
Compare three models:
- all-MiniLM-L6-v2 (small, fast)
- all-mpnet-base-v2 (medium)
- BAAI/bge-large-en-v1.5 (large, accurate)

Measure:
- Embedding time
- Memory usage
- Classification accuracy

**Deliverable:** Comparison table with recommendation

## Key Concepts

### Embeddings
Vector representations of text that capture semantic meaning. Similar texts have similar embeddings.

### Sentence Transformers
Pre-trained models that convert sentences/paragraphs into embeddings. Built on top of BERT and similar architectures.

### Cosine Similarity
Measures similarity between two vectors. Range: -1 (opposite) to 1 (identical). Used for comparing embeddings.

### Zero-shot Classification
Classify without training data by comparing document embeddings to category description embeddings.

## Recommended Models

For document classification:

1. **BAAI/bge-large-en-v1.5** - Best quality, slower
2. **BAAI/bge-base-en-v1.5** - Good balance
3. **all-mpnet-base-v2** - Alternative, widely used
4. **all-MiniLM-L6-v2** - Fastest, good for experimentation

## Tips for Success

1. **Start simple**: Use small models first (MiniLM)
2. **Visualize**: Plot embeddings to build intuition
3. **Experiment**: Try different category descriptions
4. **Iterate**: Refine category descriptions to improve accuracy

## Common Issues

### Issue: Model download is slow
**Solution:** Be patient on first run. Models are cached for future use.

### Issue: Out of memory
**Solution:** Use a smaller model (MiniLM) or process fewer documents at once.

### Issue: Low classification accuracy
**Solution:** Improve category descriptions to be more specific and detailed.

## Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Model rankings
- [Understanding Embeddings](https://vickiboykis.com/what_are_embeddings/)

## Deliverable

By end of Week 1, submit:
- Completed notebooks (run all cells)
- Comparison table of 5 models on 100 documents
- Reflection: Which model would you choose and why?

## Time Estimate

- Notebooks: 6 hours
- Exercises: 4 hours
- Experimentation: 2 hours
- **Total: ~12 hours**

## Next Week Preview

Week 2 will introduce PyTorch and build neural network classifiers that can be trained on your data. You'll learn when to use zero-shot (this week) vs trained models (next week).

---

**Ready to start?** Open `notebooks/01_what_are_embeddings.ipynb`
