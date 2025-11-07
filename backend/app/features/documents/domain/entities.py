"""Domain entities for the documents feature."""

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field

from app.shared.helpers import generate_id, current_timestamp


# Default user ID for development (no auth yet)
# Using a special UUID for system/unauthenticated users
DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000000"


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""
    
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    page_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "keywords": self.keywords,
            "page_count": self.page_count,
        }


@dataclass
class Document:
    """Core document entity."""
    
    id: str = field(default_factory=generate_id)
    user_id: str = DEFAULT_USER_ID
    course_id: str = ""  # Will be set to UUID string from DB
    filename: str = ""
    file_hash: str = ""
    total_pages: int = 0
    storage_path: str = ""
    file_size: int = 0
    mime_type: str = ""
    status: str = "pending"  # pending, processing, completed, failed
    chunks_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=current_timestamp)
    updated_at: datetime = field(default_factory=current_timestamp)
    
    
    def mark_as_processing(self) -> None:
        """Mark document as being processed."""
        self.status = "processing"
        self.updated_at = current_timestamp()
    
    def mark_as_completed(self, total_pages: int, metadata: Optional[DocumentMetadata] = None) -> None:
        """Mark document as successfully processed."""
        self.status = "completed"
        self.total_pages = total_pages
        self.metadata = metadata
        self.processed_at = current_timestamp()
        self.updated_at = current_timestamp()
    
    def mark_as_failed(self, error_message: str) -> None:
        """Mark document as failed."""
        self.status = "failed"
        self.error_message = error_message
        self.updated_at = current_timestamp()
    
    def is_processed(self) -> bool:
        """Check if document has been processed."""
        return self.status == "completed"
    
    def validate(self) -> None:
        """Validate document data."""
        from app.shared.exceptions import ValidationError
        
        if not self.user_id:
            raise ValidationError("user_id is required")
        if not self.filename:
            raise ValidationError("filename is required")
        if not self.file_hash:
            raise ValidationError("file_hash is required")
        if self.file_size_bytes <= 0:
            raise ValidationError("file_size_bytes must be positive")


@dataclass
class DocumentMarkdown:
    """Markdown representation of extracted document content."""
    
    document_id: str
    content: str  # Full markdown content
    extracted_at: datetime = field(default_factory=current_timestamp)
    
    def validate(self) -> None:
        """Validate markdown data."""
        from app.shared.exceptions import ValidationError
        
        if not self.document_id:
            raise ValidationError("document_id is required")
        if not self.content:
            raise ValidationError("content is required")
    
    def get_preview(self, max_chars: int = 500) -> str:
        """Get a preview of the markdown content."""
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + "..."


@dataclass
class DocumentChunk:
    """Represents a chunk of text extracted from a document."""
    
    id: str = field(default_factory=generate_id)
    course_id: str = ""
    document_id: str = ""
    content: str = ""
    chunk_index: int = 0
    page_number: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=current_timestamp)
    
    def validate(self) -> None:
        """Validate chunk data."""
        from app.shared.exceptions import ValidationError
        
        if not self.document_id:
            raise ValidationError("document_id is required")
        if not self.content:
            raise ValidationError("content is required")
        if self.chunk_index < 0:
            raise ValidationError("chunk_index must be non-negative")
    
    def enrich_metadata(self, document_title: Optional[str] = None) -> None:
        """Enrich chunk metadata with document information."""
        if document_title:
            self.metadata["document_title"] = document_title
        self.metadata["chunk_index"] = self.chunk_index
        self.metadata["page_number"] = self.page_number
