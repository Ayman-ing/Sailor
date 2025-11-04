"""Domain entities for the chat feature."""

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field

from app.shared.helpers import generate_id, current_timestamp


@dataclass
class RetrievedChunk:
    """Represents a document chunk retrieved from the vector store."""
    
    chunk_id: str
    document_id: str
    content: str
    score: float
    chunk_index: int = 0
    page_number: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)
    
    def get_source_info(self) -> str:
        """Get formatted source information for citations."""
        doc_title = self.metadata.get("document_title", "Unknown Document")
        if self.page_number > 0:
            return f"{doc_title} (Page {self.page_number})"
        return f"{doc_title} (Chunk {self.chunk_index})"


@dataclass
class QueryContext:
    """Context information for a query."""
    
    query: str
    user_id: str
    document_ids: Optional[List[str]] = None
    top_k: int = 5
    hybrid_alpha: float = 0.7  # Balance between dense and sparse search
    expand_context: bool = True  # Whether to retrieve neighboring chunks
    score_threshold: float = 0.7  # Minimum score to trigger context expansion
    
    def validate(self) -> None:
        """Validate query context."""
        from app.shared.exceptions import ValidationError
        
        if not self.query or not self.query.strip():
            raise ValidationError("query cannot be empty")
        if not self.user_id:
            raise ValidationError("user_id is required")
        if self.top_k <= 0:
            raise ValidationError("top_k must be positive")
        if not 0 <= self.hybrid_alpha <= 1:
            raise ValidationError("hybrid_alpha must be between 0 and 1")
        if not 0 <= self.score_threshold <= 1:
            raise ValidationError("score_threshold must be between 0 and 1")


@dataclass
class ChatMessage:
    """Represents a single message in a chat conversation."""
    
    id: str = field(default_factory=generate_id)
    role: str = "user"  # user, assistant, system
    content: str = ""
    created_at: datetime = field(default_factory=current_timestamp)
    metadata: dict = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate message."""
        from app.shared.exceptions import ValidationError
        
        if self.role not in ["user", "assistant", "system"]:
            raise ValidationError(f"Invalid role: {self.role}")
        if not self.content:
            raise ValidationError("content cannot be empty")


@dataclass
class ChatResponse:
    """Response from the chat system with context."""
    
    id: str = field(default_factory=generate_id)
    query: str = ""
    answer: str = ""
    sources: List[RetrievedChunk] = field(default_factory=list)
    model: str = ""
    created_at: datetime = field(default_factory=current_timestamp)
    metadata: dict = field(default_factory=dict)
    
    def add_source(self, chunk: RetrievedChunk) -> None:
        """Add a source chunk to the response."""
        self.sources.append(chunk)
    
    def get_formatted_sources(self) -> str:
        """Get formatted source citations."""
        if not self.sources:
            return "No sources found."
        
        citations = []
        for i, source in enumerate(self.sources, 1):
            citations.append(f"{i}. {source.get_source_info()} (score: {source.score:.3f})")
        
        return "\n".join(citations)
