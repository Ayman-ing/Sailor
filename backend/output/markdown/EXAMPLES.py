"""
Example Output Generator
This shows what the saved markdown files will look like.
"""

# Example 1: pymupdf4llm output
PYMUPDF_EXAMPLE = """# pymupdf4llm Output
Source: research_paper.pdf
Generated: 20251105_143022

---

# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence...

## Supervised Learning

In supervised learning, the algorithm learns from labeled data...

### Classification

Classification tasks involve predicting discrete labels...

### Regression

Regression tasks involve predicting continuous values...

## Unsupervised Learning

Unsupervised learning works with unlabeled data...
"""

# Example 2: Chonkie output
CHONKIE_EXAMPLE = """# Chonkie MarkdownChef Output
Source: research_paper.pdf
Generated: 20251105_143022
Total Chunks: 12
Code Blocks: 3
Tables: 2

---

## Text Chunks

### Chunk 1
**Tokens:** 145

# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

---

### Chunk 2
**Tokens:** 167

## Supervised Learning

In supervised learning, the algorithm learns from labeled data. The training dataset includes both input features and corresponding output labels. Common algorithms include:

- Linear Regression
- Decision Trees
- Neural Networks

---

### Chunk 3
**Tokens:** 198

### Classification

Classification tasks involve predicting discrete labels. For example, classifying emails as spam or not spam, or identifying handwritten digits. Popular classification algorithms include Support Vector Machines (SVM) and Random Forests.

---

## Code Blocks

### Code Block 1
```
def train_model(X, y):
    model = RandomForest()
    model.fit(X, y)
    return model
```

### Code Block 2
```
import numpy as np
from sklearn.model_selection import train_test_split
```

## Tables

### Table 1
| Algorithm | Type | Accuracy |
|-----------|------|----------|
| SVM | Classification | 94% |
| Random Forest | Classification | 92% |
| Linear Regression | Regression | 88% |
"""

if __name__ == "__main__":
    print("=" * 60)
    print("EXAMPLE: pymupdf4llm Output")
    print("=" * 60)
    print(PYMUPDF_EXAMPLE)
    print("\n\n")
    print("=" * 60)
    print("EXAMPLE: Chonkie Output")
    print("=" * 60)
    print(CHONKIE_EXAMPLE)
