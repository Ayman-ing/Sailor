"""Qdrant implementation of the EmbeddingRepository interface."""

from typing import List, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, SparseVector

from app.features.documents.domain.entities import DocumentChunk
from app.features.documents.domain.repository_interface import EmbeddingRepository
from app.core.qdrant_client import QdrantManager, generate_user_collection_name
from app.core.config import settings
from app.shared.exceptions import VectorStoreError
from app.core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingRepositoryQdrant(EmbeddingRepository):
    """
    Stores and retrieves document embeddings from a Qdrant vector database.
    This class adapts the domain's requests to Qdrant's specific API.
    """
    
    def __init__(self, qdrant_manager: QdrantManager):
        self.qdrant = qdrant_manager
        
    async def create_collection(self, user_id: str, vector_size: int) -> None:
        """Creates a Qdrant collection for the user's document embeddings."""
        collection_name = generate_user_collection_name(user_id)
        client = self.qdrant.get_client()
        await self._ensure_collection_exists(client, collection_name)
        
    async def _ensure_collection_exists(self, client: QdrantClient, collection_name: str):
        """Creates a Qdrant collection if it doesn't exist, configured for hybrid search."""
        try:
            client.get_collection(collection_name=collection_name)
            logger.debug(f"Collection '{collection_name}' already exists.")
        except Exception:
            logger.info(f"Collection '{collection_name}' not found. Creating it for hybrid search...")
            # Using named vectors configuration
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text-dense": models.VectorParams(
                        size=settings.embedding_dim, 
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams()
                }
            )
            logger.info(f"Collection '{collection_name}' created successfully.")

    async def store_chunks(
        self,
        user_id: str,
        document_id: str,
        chunks: List[DocumentChunk],
        dense_embeddings: List[List[float]],
        sparse_embeddings: List[SparseVector],
        batch_size: int = 50
    ) -> List[str]:
        """
        Upserts document chunks with both dense and sparse vectors in batches.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            chunks: List of document chunks
            dense_embeddings: Dense vector embeddings
            sparse_embeddings: Sparse vector embeddings
            batch_size: Number of points to upsert per batch (default: 50)
        
        Returns:
            List of chunk IDs that were stored
        """
        collection_name = generate_user_collection_name(user_id)
        client = self.qdrant.get_client()
        await self._ensure_collection_exists(client, collection_name)

        if not chunks:
            logger.warning("No points to upsert. Skipping storage.")
            return []

        chunk_ids = []
        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        logger.info(f"Upserting {total_chunks} points in {total_batches} batches of {batch_size}...")

        try:
            # Process in batches
            for batch_idx in range(0, total_chunks, batch_size):
                batch_end = min(batch_idx + batch_size, total_chunks)
                current_batch_num = batch_idx // batch_size + 1
                
                logger.info(f"Processing batch {current_batch_num}/{total_batches} ({batch_end - batch_idx} points)...")
                
                # Prepare batch data
                batch_chunks = chunks[batch_idx:batch_end]
                batch_dense = dense_embeddings[batch_idx:batch_end]
                batch_sparse = sparse_embeddings[batch_idx:batch_end]
                
                points_to_upsert: List[PointStruct] = []
                
                for i, chunk in enumerate(batch_chunks):
                    point_id = chunk.id
                    chunk_ids.append(point_id)
                    
                    payload = {
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "metadata": chunk.metadata,
                    }
                    
                    # Create point with named vectors (both dense and sparse)
                    points_to_upsert.append(
                        PointStruct(
                            id=point_id,
                            vector={
                                "text-dense": batch_dense[i],
                                "text-sparse": {
                                    "indices": batch_sparse[i].indices,
                                    "values": batch_sparse[i].values,
                                }
                            },
                            payload=payload
                        )
                    )
                
                # Upsert the batch
                client.upsert(
                    collection_name=collection_name,
                    points=points_to_upsert,
                    wait=True
                )
                
                logger.info(f"Batch {current_batch_num}/{total_batches} upserted successfully.")
            
            logger.info(f"All {total_chunks} points upserted successfully.")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant: {e}", exc_info=True)
            raise VectorStoreError(f"Could not store chunks in Qdrant: {str(e)}")

    async def delete_document_chunks(self, user_id: str, document_id: str) -> None:
        collection_name = generate_user_collection_name(user_id)
        
        if not self.qdrant.get_client().collection_exists(collection_name):
            logger.warning(f"Attempted to delete chunks from non-existent collection: {collection_name}")
            return
            
        logger.info(f"Deleting all chunks for document_id '{document_id}' from collection '{collection_name}'.")
        
        self.qdrant.get_client().delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True
        )

    async def search_similar(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[dict]:
        collection_name = generate_user_collection_name(user_id)
        
        if not self.qdrant.get_client().collection_exists(collection_name):
            logger.warning(f"Attempted to search in non-existent collection: {collection_name}")
            return []
            
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
        
        # Search using the named dense vector
        search_result = self.qdrant.get_client().search(
            collection_name=collection_name,
            query_vector=("text-dense", query_embedding),  # Named vector query
            limit=top_k,
            query_filter=search_filter
        )
        
        return [hit.model_dump() for hit in search_result]