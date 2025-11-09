"""SQLAlchemy models for documents feature."""

from datetime import datetime

from sqlalchemy import Column, String, Integer,UUID, DateTime, Text, JSON

from app.core.database import Base


class DocumentModel(Base):
    """Document table."""
    __tablename__ = "documents"
    
    id = Column(UUID, primary_key=True)
    user_id = Column(UUID, nullable=False, index=True)
    course_id = Column(UUID, nullable=False, index=True)
    filename = Column(String, nullable=False)
    file_hash = Column(String, nullable=False, index=True)
    file_size = Column(Integer, nullable=False)
    total_pages = Column(Integer, default=0)
    status = Column(String, default="pending", index=True)
    storage_path = Column(String, nullable=False)
    chunks_count = Column(Integer, default=0)    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"


class CourseModel (Base):
    """Course table."""
    __tablename__ = "courses"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Course(id={self.id}, title={self.title})>"