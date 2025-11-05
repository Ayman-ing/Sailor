# 🧭 Sailor - AI-Powered Study Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> An intelligent AI assistant that helps students navigate their course materials through RAG-powered conversations.

## 📖 Overview

**Sailor** allows students to:
- 📄 Upload course PDFs and documents
- 🔍 Extract and index content using advanced NLP
- 💬 Chat with an AI assistant that understands their course materials
- 🎯 Get accurate, context-aware answers from their documents

## ✨ Features

- **Document Processing**: Upload and parse PDFs using PyMuPDF4LLM and Chonkie
- **Smart Indexing**: Vector embeddings stored in Qdrant for fast retrieval
- **Hybrid Search**: Dense + sparse embeddings with RRF fusion
- **Context Expansion**: Automatic retrieval of neighboring chunks for better context
- **RAG Pipeline**: Retrieval-Augmented Generation powered by Groq LLM
- **Fast Inference**: Powered by Groq API (LLaMA3 models)
- **Clean Architecture**: Feature-based modular design
- **Type Safety**: Full Pydantic validation throughout

## 🏗️ Architecture

```
Sailor/
├── backend/          # FastAPI + RAG Pipeline
│   ├── app/
│   │   ├── core/           # Configuration, DB, Qdrant
│   │   ├── features/       # Documents, Chat, Users
│   │   └── shared/         # Common utilities
│   └── docker-compose.yml
│
└── frontend/         # (Coming soon)
```

**Architecture Pattern**: Clean Architecture with Domain-Driven Design
- 📘 **Domain Layer**: Pure business logic
- 🔧 **Application Layer**: Use cases
- 🗄️ **Infrastructure Layer**: Database, APIs, services
- 🌐 **Presentation Layer**: FastAPI routes

See [`backend/CLAUDE.md`](backend/CLAUDE.md) for detailed architecture documentation.

## 🚀 Tech Stack

### Backend
| Category | Technology |
|----------|------------|
| Framework | FastAPI |
| Document Parsing | PyMuPDF4LLM, Docling |
| Chunking | Chonkie (MarkdownChef) |
| Embeddings | SentenceTransformers (dense), SPLADE (sparse) |
| Vector DB | Qdrant |
| Relational DB | PostgreSQL |
| LLM API | Groq (LLaMA3) |
| Validation | Pydantic |

### DevOps
- **Containerization**: Docker + Docker Compose
- **Package Manager**: UV (Python)

## 📦 Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- [UV](https://github.com/astral-sh/uv) package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Sailor.git
   cd Sailor
   ```

2. **Backend setup**
   ```bash
   cd backend
   
   # Install dependencies
   uv sync
   
   # Create .env from template
   cp .env.example .env
   
   # Edit .env and add your API keys
   # - GROQ_API_KEY (get from https://console.groq.com/keys)
   # - SECRET_KEY (generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
   ```

3. **Start services**
   ```bash
   # Start PostgreSQL and Qdrant
   docker-compose up -d
   
   # Run the application
   uv run uvicorn app.main:app --reload
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Qdrant Dashboard: http://localhost:6333/dashboard

## 🔧 Configuration

All configuration is in `backend/.env`. Key variables:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_secret_key_here

# Optional (defaults provided)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/sailor_db
QDRANT_URL=http://localhost:6333
GROQ_MODEL=llama3-70b-8192
```

See [`backend/.env.example`](backend/.env.example) for all options.

## 📚 Usage

### Upload a Document
```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

### Chat with Your Documents
```bash
curl -X POST "http://localhost:8000/api/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key concepts in chapter 3?",
    "document_id": "uuid-here"
  }'
```

## 🧪 Testing

```bash
cd backend

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run specific test
uv run pytest tests/unit/features/documents/
```

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py                 # Application entry point
│   ├── api_router.py           # Main API router
│   │
│   ├── core/                   # Core configuration
│   │   ├── config.py
│   │   ├── database.py
│   │   └── qdrant_client.py
│   │
│   ├── features/               # Feature modules
│   │   ├── documents/          # Document management
│   │   │   ├── domain/
│   │   │   ├── application/
│   │   │   ├── infrastructure/
│   │   │   └── presentation/
│   │   │
│   │   └── chat/               # Chat functionality
│   │       ├── domain/
│   │       ├── application/
│   │       ├── infrastructure/
│   │       └── presentation/
│   │
│   └── shared/                 # Shared utilities
│
├── tests/                      # Test suite
├── migrations/                 # Database migrations
└── docker-compose.yml
```

## 🛣️ Roadmap

- [x] Project structure and architecture
- [x] Document upload and processing
- [x] Vector embedding and storage (hybrid search)
- [x] RAG-powered chat interface
- [x] Context expansion for better answers
- [ ] User authentication
- [ ] Document management (list, delete)
- [ ] Chat history
- [ ] Frontend UI
- [ ] Deployment configuration

## 🤝 Contributing

Contributions are welcome! Please read our architecture guidelines in [`backend/CLAUDE.md`](backend/CLAUDE.md) before contributing.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Follow the Clean Architecture pattern
4. Write tests for your code
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Chonkie](https://github.com/bhavnicksm/chonkie) - Intelligent text chunking
- [PyMuPDF4LLM](https://github.com/pymupdf/PyMuPDF4LLM) - PDF to markdown conversion
- [Docling](https://github.com/DS4SD/docling) - Document understanding and parsing
- [Qdrant](https://qdrant.tech/) - Vector database
- [SentenceTransformers](https://www.sbert.net/) - Dense embeddings
- [SPLADE](https://github.com/naver/splade) - Sparse embeddings
- [Groq](https://groq.com/) - Fast LLM inference

## 📧 Contact

Project Link: [https://github.com/yourusername/Sailor](https://github.com/yourusername/Sailor)

---

Built with ❤️ for students everywhere