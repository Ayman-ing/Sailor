"""SQLAlchemy models for documents feature."""

from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON

from app.core.database import Base


class DocumentModel(Base):
    """Document table."""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    filename = Column(String, nullable=False)
    file_hash = Column(String, nullable=False, index=True)
    file_size_bytes = Column(Integer, nullable=False)
    total_pages = Column(Integer, default=0)
    status = Column(String, default="pending", index=True)
    
    # Metadata stored as JSON
    metadata = Column(JSON, nullable=True)
    
    # Chunking info
    chunks_count = Column(Integer, default=0)
    chunks_stored_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"


class DocumentMarkdownModel(Base):
    """Document markdown content table."""
    __tablename__ = "document_markdowns"
    
    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False, unique=True, index=True)
    content = Column(Text, nullable=False)
    extracted_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<DocumentMarkdown(document_id={self.document_id})>"
