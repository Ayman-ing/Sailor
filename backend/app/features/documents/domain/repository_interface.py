"""Repository interfaces for the documents feature."""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client.http.models import SparseVector

from app.features.documents.domain.entities import Document, DocumentChunk
from app.features.documents.domain.value_objects import DocumentFilter


class DocumentRepository(ABC):
    """Interface for document persistence."""
    
    @abstractmethod
    async def save(self, document: Document) -> Document:
        """Save a document (create or update)."""
        pass
    
    @abstractmethod
    async def get_by_id(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        pass
    
    @abstractmethod
    async def get_by_hash(self, file_hash: str, user_id: str) -> Optional[Document]:
        """Get document by file hash for a specific user."""
        pass
    
    @abstractmethod
    async def delete(self, document_id: str) -> None:
        """Delete a document."""
        pass
    
    @abstractmethod
    async def list_by_filter(self, filter: DocumentFilter) -> List[Document]:
        """List documents matching filter criteria."""
        pass


class EmbeddingRepository(ABC):
    """Interface for vector embeddings storage."""
    
    @abstractmethod
    async def create_collection(self, user_id: str, vector_size: int) -> None:
        """Create a collection for user's document embeddings."""
        pass
    
    @abstractmethod
    async def store_chunks(
        self, 
        user_id: str,
        document_id: str,
        chunks: List[DocumentChunk],
        dense_embeddings: List[List[float]],
        sparse_embeddings: List["SparseVector"]
    ) -> List[str]:
        """Store document chunks with their dense and sparse embeddings."""
        pass
    
    @abstractmethod
    async def delete_document_chunks(self, user_id: str, document_id: str) -> None:
        """Delete all chunks for a specific document."""
        pass
    
    @abstractmethod
    async def search_similar(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int,
        document_ids: Optional[List[str]] = None
    ) -> List[dict]:
        """Search for similar chunks."""
        pass
