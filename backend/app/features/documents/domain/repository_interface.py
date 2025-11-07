"""Repository interfaces for the documents feature."""

from abc import ABC, abstractmethod
from typing import List, Optional

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
    """Interface for vector embeddings storage with hybrid search support."""
    
    @abstractmethod
    async def create_collection(self, user_id: str, vector_size: int) -> None:
        """Create a collection for user's document embeddings."""
        pass
    
    @abstractmethod
    async def store_chunks(
        self, 
        user_id: str,
        course_id: str,
        chunks: List[DocumentChunk],
        dense_embeddings: List[List[float]],
        sparse_embeddings: List,  # List[SparseVector] - avoiding import here
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Store document chunks with both dense and sparse embeddings.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            chunks: List of document chunks
            dense_embeddings: Dense vector embeddings
            sparse_embeddings: Sparse vector embeddings (SPLADE)
            batch_size: Optional batch size for processing
            
        Returns:
            List of chunk IDs that were stored
        """
        pass
    
    @abstractmethod
    async def delete_document_chunks(self, user_id: str, document_id: str) -> None:
        """Delete all chunks for a specific document."""
        pass
    
    @abstractmethod
    async def search_similar(
        self,
        user_id: str,
        query_dense_embedding: List[float],
        query_sparse_embedding: Optional[any] = None,  # Optional[SparseVector]
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        hybrid_alpha: float = 0.5
    ) -> List[dict]:
        """
        Search for similar chunks using hybrid search (dense + sparse).
        
        Args:
            user_id: User identifier
            query_dense_embedding: Dense vector embedding of the query
            query_sparse_embedding: Sparse vector embedding (optional)
            top_k: Number of results to return
            document_ids: Optional list of document IDs to filter by
            hybrid_alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
            
        Returns:
            List of search results with scores and payloads
        """
        pass
