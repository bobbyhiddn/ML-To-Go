# Sample Data

This directory contains sample files for testing your extraction and classification pipelines.

## Structure

```
sample-data/
├── invoices/       # Sample invoices (PDF, XLSX)
├── contracts/      # Sample contracts (DOCX, PDF)
├── reports/        # Sample reports (XLSX, PDF)
├── emails/         # Sample emails (TXT, EML)
└── mixed/          # Mixed file types for testing
```

## Adding Your Own Samples

1. Collect diverse document types from your work
2. Remove any sensitive/confidential information
3. Place in appropriate category folders
4. Use for testing throughout weeks 4-6

## Pre-populated Samples

The repository includes a few anonymized sample files to get started:
- Invoice template (PDF)
- Contract template (DOCX)
- Financial report (XLSX)
- Email thread (TXT)

Add more as you progress through the training.

## Usage

### Week 4: Text Extraction
Test your extractors on these sample files:
```python
from week4_text_extraction.extractors import unified

extractor = unified.UnifiedExtractor()
text = extractor.extract('sample-data/invoices/invoice001.pdf')
```

### Week 5: Classification
Use for testing your complete pipeline:
```python
from week5_classification_engine.classifier import Classifier

classifier = Classifier()
results = classifier.classify_directory('sample-data/')
```

## File Formats Supported

- **PDF:** `.pdf`
- **Word:** `.docx`, `.doc`
- **Excel:** `.xlsx`, `.xls`
- **PowerPoint:** `.pptx`, `.ppt`
- **Text:** `.txt`
- **Email:** `.eml`, `.msg`
- **Images:** `.jpg`, `.png` (with OCR)

## Data Privacy

- Never commit real customer or sensitive data
- Use anonymized or synthetic documents only
- Review files before committing

## Generating Synthetic Data

For testing at scale, consider:
1. Creating templates with placeholder data
2. Using faker library to generate text
3. Duplicating and modifying existing samples

Example:
```python
from faker import Faker
fake = Faker()

# Generate fake invoice text
invoice_text = f"""
Invoice #{fake.random_number(digits=6)}
Date: {fake.date()}
Customer: {fake.company()}
Amount: ${fake.random_number(digits=4)}
"""
```

## Best Practices

1. **Organize by category** - Helps with testing classification
2. **Include edge cases** - Encrypted PDFs, scanned documents, corrupt files
3. **Vary formats** - Different PDF types, Office versions
4. **Document sources** - Keep notes on where samples came from

## .gitignore Configuration

The `.gitignore` is configured to:
- Ignore all files in subdirectories (`sample-data/*/`)
- Keep this README.md

To commit sample files, you'll need to force add them:
```bash
git add -f sample-data/invoices/sample.pdf
```

## Need More Sample Documents?

Public datasets for document classification:
- [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) - 400k document images
- [Tobacco3482](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg) - Industry documents
- Create your own using templates

---

**Ready to start?** Add your first sample file and test extraction in Week 4.
