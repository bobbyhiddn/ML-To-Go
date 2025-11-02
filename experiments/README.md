# Experiments Folder

Use this folder for quick experiments and prototyping with Claude Code.

## Purpose

This folder is your scratchpad for:
- Testing new ideas quickly
- Comparing approaches
- Prototyping before adding to main codebase
- Learning new libraries

## Usage with Claude Code

```bash
cd experiments
claude-code

"Create a script to compare BGE-large vs E5-large on 50 documents"
```

Claude Code will generate experiments here without cluttering your main codebase.

## .gitignore Configuration

Python files (`.py`) in this directory are ignored by git by default. Only this README and explicitly added files are tracked.

## Best Practices

1. **Name files descriptively:** `compare_embeddings_2024_01_15.py`
2. **Add comments:** Explain what you're testing
3. **Copy successful experiments:** Move to appropriate week folders
4. **Clean up regularly:** Delete old experiments

## Example Experiments

### Week 1: Compare Embedding Models
```python
# experiments/embedding_comparison.py
from sentence_transformers import SentenceTransformer

models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'BAAI/bge-large-en-v1.5']
texts = ["Sample document text..."]

for model_name in models:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    print(f"{model_name}: {embeddings.shape}")
```

### Week 2: Test PyTorch Architecture
```python
# experiments/test_architecture.py
import torch.nn as nn

class TestClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = TestClassifier()
print(model)
```

## Tips

- Use experiments to validate ideas before implementing in main code
- Test with small datasets first
- Document successful approaches for future reference

---

**Start experimenting!** This is your sandbox - try anything.
