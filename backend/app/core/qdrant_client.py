from typing import List, Optional, Dict, Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class QdrantManager:
    """Manages Qdrant vector database operations."""
    
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    
    async def create_collection(
        self, 
        collection_name: str, 
        vector_size: int = 384,
        distance: Distance = Distance.COSINE
    ) -> None:
        """Create a new collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
                logger.info(f"Collection {collection_name} created successfully")
            else:
                logger.info(f"Collection {collection_name} already exists")
        
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection {collection_name}: {e}")
            raise
    
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return collection_name in collection_names
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    async def add_points(
        self, 
        collection_name: str,
        vectors: List[List[float]], 
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors to the specified collection."""
        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in vectors]
            
            points = [
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ]
            
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Added {len(points)} points to collection {collection_name}")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add points to Qdrant collection {collection_name}: {e}")
            raise
    
    async def search_similar(
        self, 
        collection_name: str,
        query_vector: List[float], 
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the specified collection."""
        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for point in search_result:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })
            
            logger.info(f"Found {len(results)} similar vectors in {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search in Qdrant collection {collection_name}: {e}")
            raise
    
    async def delete_points(self, collection_name: str, ids: List[str]) -> None:
        """Delete points from the specified collection."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=ids)
            )
            logger.info(f"Deleted {len(ids)} points from collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete points from Qdrant collection {collection_name}: {e}")
            raise
    
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection {collection_name} deleted successfully")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            raise
    
    def close(self) -> None:
        """Close the Qdrant client."""
        self.client.close()


# Global Qdrant manager instance
qdrant_manager = QdrantManager()


# Helper functions to generate collection names
def generate_user_collection_name(user_id: str) -> str:
    """Generate a collection name for all user documents."""
    return f"user_{user_id}_documents"


def generate_document_collection_name(user_id: str, document_id: str) -> str:
    """Generate a collection name for a specific document."""
    return f"user_{user_id}_doc_{document_id}"