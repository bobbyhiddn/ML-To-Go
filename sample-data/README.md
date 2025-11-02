# Sample Data

This directory is for storing sample documents and datasets for training and testing.

## Recommended Structure
- `raw/` - Raw, unprocessed documents
- `processed/` - Cleaned and preprocessed data
- `train/` - Training dataset
- `val/` - Validation dataset
- `test/` - Test dataset

## File Types
- PDF documents (`.pdf`)
- Word documents (`.docx`)
- Text files (`.txt`)
- CSV files for labels and metadata

## Note
Large data files are ignored by git. Consider using:
- Git LFS for version controlling large files
- Cloud storage (S3) for production datasets
- Public datasets from Kaggle, Hugging Face, etc.

## Sample Datasets
For document classification, consider:
- 20 Newsgroups
- Reuters-21578
- AG News
- DBpedia
