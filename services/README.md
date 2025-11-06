# Sailor Embedding Services

This directory contains microservices for generating embeddings used by the Sailor RAG system.

## Services

### 🧠 Dense Embedding Service (`dense-embedding/`)
- **Port**: 8001
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Output**: Dense vectors (384 dimensions)
- **Use**: Semantic similarity search

### 🔍 Sparse Embedding Service (`sparse-embedding/`)
- **Port**: 8002
- **Model**: `prithvida/Splade_PP_en_v1`
- **Output**: Sparse vectors (TF-IDF like)
- **Use**: Keyword/lexical search

## Quick Start

### Run All Services
From the root directory:
```bash
./dev.sh
```

This will start:
- Dense Embedding Service on http://localhost:8001
- Sparse Embedding Service on http://localhost:8002
- Main Backend on http://localhost:8000

### Run Individual Services

**Dense Embedding:**
```bash
cd services/dense-embedding
uv pip install -r requirements.txt
uv run python app.py
```

**Sparse Embedding:**
```bash
cd services/sparse-embedding
uv pip install -r requirements.txt
uv run python app.py
```

## API Usage

### Dense Embeddings
```bash
curl -X POST "http://localhost:8001/embed" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "How are you?"]}'
```

### Sparse Embeddings
```bash
curl -X POST "http://localhost:8002/embed" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "How are you?"]}'
```

## Health Checks
- Dense: http://localhost:8001/health
- Sparse: http://localhost:8002/health

## Documentation
- Dense: http://localhost:8001/docs
- Sparse: http://localhost:8002/docs
