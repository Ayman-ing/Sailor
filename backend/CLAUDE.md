# 🧭 CLAUDE.md — Sailor Architecture & Guidelines

## 📘 Project Overview

**Sailor** is an AI-powered assistant for students.  
It allows users to:
- Upload their **course PDFs**
- Extract and index the content using **PyMuPDF4LLM** + **LlamaIndex**
- Store embeddings in **Qdrant**
- Chat with a **retrieval-augmented chatbot** powered by **Groq API** (LLaMA3 or other open models)

This document defines the **architecture principles**, **folder structure**, and **coding rules** that all contributors (and AI tools like GitHub Copilot) must follow.

---

## 🧱 Architecture Philosophy

Sailor uses a **Clean Architecture** with **feature-based modularization**.

Each feature (e.g., `documents`, `chat`, `users`) contains:
- `domain/` — Core business entities & interfaces
- `application/` — Use cases (business logic)
- `infrastructure/` — Framework & service implementations
- `presentation/` — FastAPI routes, schemas, controllers

We strictly follow **dependency inversion**:
> Inner layers (domain/application) must not depend on outer layers (infrastructure/presentation).

---

## 🧩 Layer Responsibilities

### 1️⃣ Domain Layer
- Pure business logic.
- Contains **entities**, **value objects**, and **repository interfaces**.
- No external dependencies (no FastAPI, DB, etc.).

### 2️⃣ Application Layer
- Implements **use cases** (the core business flows).
- Coordinates domain entities and repository interfaces.
- Contains no technical details (no DB calls, no HTTP).

### 3️⃣ Infrastructure Layer
- Adapters for databases, APIs, and frameworks.
- Implements repository interfaces (e.g., Postgres, Qdrant, Groq).
- Handles I/O, network, and persistence.

### 4️⃣ Presentation Layer
- FastAPI routes and request/response models.
- Maps user input → use case execution → output serialization.

---

## 📂 Project Structure
```
sailor/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── qdrant_client.py
│   │   └── logger.py
│   │
│   ├── features/
│   │   ├── documents/
│   │   │   ├── domain/
│   │   │   │   ├── entities.py
│   │   │   │   ├── repository_interface.py
│   │   │   │   └── value_objects.py
│   │   │   ├── application/
│   │   │   │   ├── upload_document_usecase.py
│   │   │   │   ├── extract_chunks_usecase.py
│   │   │   │   └── index_document_usecase.py
│   │   │   ├── infrastructure/
│   │   │   │   ├── pdf_extractor_pymupdf.py
│   │   │   │   ├── document_repository_pg.py
│   │   │   │   └── embedding_repository_qdrant.py
│   │   │   ├── presentation/
│   │   │   │   ├── routes.py
│   │   │   │   └── schemas.py
│   │   │   └── __init__.py
│   │   │
│   │   ├── chat/
│   │   │   ├── domain/
│   │   │   │   ├── entities.py
│   │   │   │   ├── repository_interface.py
│   │   │   ├── application/
│   │   │   │   ├── query_documents_usecase.py
│   │   │   │   └── chat_with_context_usecase.py
│   │   │   ├── infrastructure/
│   │   │   │   ├── llm_groq_service.py
│   │   │   │   └── retriever_qdrant.py
│   │   │   ├── presentation/
│   │   │   │   ├── routes.py
│   │   │   │   └── schemas.py
│   │   │   └── __init__.py
│   │   │
│   │   └── users/
│   │       ├── domain/
│   │       ├── application/
│   │       ├── infrastructure/
│   │       ├── presentation/
│   │       └── __init__.py
│   │
│   ├── shared/
│   │   ├── exceptions.py
│   │   ├── helpers.py
│   │   ├── interfaces.py
│   │   └── utils/
│   │       └── chunking.py
│   │
│   └── api_router.py
│
├── docker-compose.yml
├── pyproject.toml
└── README.md

```
---

## 🧠 Core Technologies

| Category | Technology |
|-----------|-------------|
| Framework | **FastAPI** |
| Document Parsing | **PyMuPDF4LLM** |
| RAG Engine | **LlamaIndex** (open-source only) |
| Vector Database | **Qdrant** |
| Relational DB | **PostgreSQL** |
| LLM API | **Groq API** |
| Schema Validation | **Pydantic** |
| Containerization | **Docker + Compose** |

---

## 🧩 Coding Guidelines

### ✅ General Rules
- Use **async/await** everywhere possible.
- Use **type hints** and **Pydantic** models for all I/O.
- Never call database or API clients directly from routes — always use **use cases**.
- No business logic in routes or repository classes.
- Follow **SOLID** principles.

### 🧩 Naming Conventions
| Type | Convention | Example |
|------|-------------|----------|
| Use Case | `verb_noun_usecase.py` | `upload_document_usecase.py` |
| Entity | PascalCase | `Document`, `Chunk`, `User` |
| Repository Interface | `SomethingRepository` | `DocumentRepository` |
| Infrastructure Impl | `*_pg.py`, `*_qdrant.py` | `document_repository_pg.py` |
| Route Files | `routes.py` | — |
| Pydantic Models | `SomethingSchema` | `UploadDocumentSchema` |

### 🧩 Testing Rules
- Unit tests for each use case (mock external deps)
- Integration tests for major feature flows
- Avoid testing 3rd-party libs directly

---