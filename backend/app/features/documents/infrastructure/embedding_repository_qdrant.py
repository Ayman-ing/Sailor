"""Qdrant implementation of the EmbeddingRepository interface."""

from typing import List, Optional
from qdrant_client import models

from app.features.documents.domain.entities import DocumentChunk
from app.features.documents.domain.repository_interface import EmbeddingRepository
from app.core.qdrant_client import QdrantManager, generate_user_collection_name
from app.core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingRepositoryQdrant(EmbeddingRepository):
    """
    Stores and retrieves document embeddings from a Qdrant vector database.
    This class adapts the domain's requests to Qdrant's specific API.
    """
    
    def __init__(self, qdrant_manager: QdrantManager):
        """
        Initializes the repository with a QdrantManager instance.
        
        Args:
            qdrant_manager: An instance of the QdrantManager to handle DB connections.
        """
        self.qdrant = qdrant_manager
    async def create_collection(self, user_id: str, vector_size: int) -> None:
        """
        Creates a Qdrant collection for storing a user's document embeddings.
        
        Args:
            user_id: The ID of the user.
            vector_size: The dimensionality of the embeddings to be stored.
        """
        collection_name = generate_user_collection_name(user_id)
        await self.qdrant.create_collection(collection_name, vector_size)
        
    async def store_chunks(
        self,
        user_id: str,
        document_id: str,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Upserts document chunks and their embeddings into a user-specific collection.
        
        Args:
            user_id: The ID of the user.
            document_id: The ID of the document.
            chunks: A list of DocumentChunk domain entities.
            embeddings: A list of corresponding vector embeddings.
            
        Returns:
            A list of the IDs of the stored chunks.
        """
        collection_name = generate_user_collection_name(user_id)
        
        # Ensure the collection exists before trying to add points.
        # The vector size is determined from the first embedding.
        if embeddings:
            vector_size = len(embeddings[0])
            await self.qdrant.create_collection(collection_name, vector_size)
        
        # Prepare the payload for each vector point. This is the metadata
        # that gets stored alongside the vector in Qdrant.
        payloads = [
            {
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        
        # Delegate the actual upsert operation to the QdrantManager.
        chunk_ids = await self.qdrant.add_points(
            collection_name=collection_name,
            vectors=embeddings,
            payloads=payloads,
            ids=[chunk.id for chunk in chunks]
        )
        
        return chunk_ids

    async def delete_document_chunks(self, user_id: str, document_id: str) -> None:
        """
        Deletes all vector points associated with a specific document ID.
        
        Args:
            user_id: The ID of the user whose collection to search.
            document_id: The ID of the document whose chunks should be deleted.
        """
        collection_name = generate_user_collection_name(user_id)
        
        if not await self.qdrant.collection_exists(collection_name):
            logger.warning(f"Attempted to delete chunks from non-existent collection: {collection_name}")
            return
            
        logger.info(f"Deleting all chunks for document_id '{document_id}' from collection '{collection_name}'.")
        
        # Use Qdrant's filtering capability to delete points where the
        # payload's 'document_id' field matches.
        await self.qdrant.delete_points_by_filter(
            collection_name=collection_name,
            points_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        )

    async def search_similar(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[dict]:
        """
        Searches for chunks with embeddings similar to the query embedding.
        
        Args:
            user_id: The user's collection to search within.
            query_embedding: The vector embedding of the search query.
            top_k: The number of similar results to return.
            document_ids: An optional list of document IDs to restrict the search to.
            
        Returns:
            A list of dictionaries, each representing a found chunk with its score.
        """
        collection_name = generate_user_collection_name(user_id)
        
        if not await self.qdrant.collection_exists(collection_name):
            logger.warning(f"Attempted to search in non-existent collection: {collection_name}")
            return []
            
        # Build a filter if specific document IDs are provided
        search_filter = None
        if document_ids:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=document_ids),
                    )
                ]
            )
        
        # Delegate the search operation to the QdrantManager.
        results = await self.qdrant.search_similar(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            filter_conditions=search_filter
        )
        
        return results