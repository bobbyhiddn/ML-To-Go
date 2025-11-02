# Local Setup Guide

## System Requirements

- Python 3.11 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Internet connection for downloading models

## GitHub Codespaces Setup (Recommended)

1. **Fork or clone this repository**
2. **Click "Code" → "Codespaces" → "Create codespace on main"**
3. **Wait for container to build** (5-10 minutes first time)
4. **Verify installation:**
   ```bash
   python --version
   pip list | grep torch
   pip list | grep tensorflow
   ```

### Codespace Configuration

The `.devcontainer/devcontainer.json` automatically:
- Installs Python 3.11
- Installs all dependencies from requirements.txt
- Configures Jupyter
- Sets up port forwarding for Jupyter (8888), TensorBoard (6006), and Streamlit (8501)
- Mounts your AWS credentials (if available)

## Local Development Setup

If you prefer to run locally instead of Codespaces:

### 1. Install Python 3.11

**macOS:**
```bash
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

**Windows:**
Download from python.org

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/ml-classification-training.git
cd ml-classification-training
```

### 3. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"
python -c "import sentence_transformers; print('Sentence Transformers: OK')"
```

## Jupyter Setup

### Start Jupyter Notebook

```bash
jupyter notebook
```

This will open Jupyter in your browser at http://localhost:8888

### Start Jupyter Lab (Alternative)

```bash
jupyter lab
```

### Configure Jupyter Kernel

If Jupyter doesn't recognize your virtual environment:

```bash
python -m ipykernel install --user --name=ml-classification --display-name "Python (ML Classification)"
```

## Common Issues

### Issue: PyTorch CUDA not available

**Solution:** This repository uses CPU versions for Codespaces compatibility. For GPU support:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: TensorFlow import error

**Solution:** Ensure you have Python 3.11 (not 3.12):
```bash
python --version
```

### Issue: Jupyter kernel dies when loading large models

**Solution:** Increase available memory or use smaller models for initial learning.

### Issue: sentence-transformers model download fails

**Solution:** Set HuggingFace cache directory:
```bash
export TRANSFORMERS_CACHE=/path/with/space
export SENTENCE_TRANSFORMERS_HOME=/path/with/space
```

## AWS Configuration (Weeks 7-8 Only)

### Install AWS CLI

**macOS/Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**Windows:**
Download from https://aws.amazon.com/cli/

### Configure AWS Credentials

```bash
aws configure
```

Enter:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., us-east-1)
- Default output format (json)

### Verify AWS Configuration

```bash
aws sts get-caller-identity
```

## Disk Space Management

Models and data can consume significant disk space:

### Check Current Usage

```bash
du -sh ~/.cache/huggingface
du -sh sample-data/
```

### Clear Cached Models

```bash
rm -rf ~/.cache/huggingface/hub
```

Models will re-download when needed.

## Performance Tips

1. **Use smaller models for initial learning** (all-MiniLM-L6-v2 instead of bge-large)
2. **Close unused notebooks** to free memory
3. **Use batch processing** for large datasets
4. **Enable quantization** (Week 6) for faster inference

## Next Steps

Once setup is complete:
1. Open `week1-embeddings/notebooks/01_what_are_embeddings.ipynb`
2. Run through the notebook
3. Proceed to exercises

## Getting Help

- Check GitHub Issues
- Review `docs/weekly-objectives.md`
- Use Claude Code for specific setup questions
