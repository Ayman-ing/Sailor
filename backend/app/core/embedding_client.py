"""HTTP client for embedding services."""

import httpx
import asyncio
from typing import List
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class SparseVector:
    """Sparse vector representation matching Qdrant's format."""
    def __init__(self, indices: List[int], values: List[float]):
        self.indices = indices
        self.values = values


class EmbeddingClient:
    
    def __init__(self):
        self.dense_url = settings.dense_embedding_url
        self.sparse_url = settings.sparse_embedding_url
        self.timeout = 60.0
        self.max_retries = settings.embedding_max_retries
        self.retry_delay = settings.embedding_retry_delay
        
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0
            )
        )
    
    async def _retry_request(self, request_func, service_name: str):
        """Retry logic with exponential backoff."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await request_func()
            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"{service_name} request failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{service_name} request failed after {self.max_retries} attempts: {e}")
            except Exception as e:
                logger.error(f"{service_name} error: {e}")
                raise
        
        raise Exception(f"Failed to connect to {service_name} after {self.max_retries} attempts: {str(last_error)}")
    
    async def get_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get dense embeddings with order preservation via indexed requests."""
        async def request():
            indexed_texts = [{"index": i, "text": text} for i, text in enumerate(texts)]
            response = await self._client.post(
                f"{self.dense_url}/embed",
                json={"texts": indexed_texts}
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract and re-sort by index to preserve order
            embeddings_with_index = [
                (emb["index"], emb["embedding"]) for emb in data["embeddings"]
            ]
            embeddings_with_index.sort(key=lambda x: x[0])
            
            return [emb for _, emb in embeddings_with_index]
        
        return await self._retry_request(request, "Dense embedding service")
    
    async def get_sparse_embeddings(self, texts: List[str]) -> List[SparseVector]:
        """Get sparse embeddings with order preservation via indexed requests."""
        async def request():
            indexed_texts = [{"index": i, "text": text} for i, text in enumerate(texts)]
            response = await self._client.post(
                f"{self.sparse_url}/embed",
                json={"texts": indexed_texts}
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract and re-sort by index to preserve order
            embeddings_with_index = [
                (emb["index"], SparseVector(indices=emb["indices"], values=emb["values"]))
                for emb in data["embeddings"]
            ]
            embeddings_with_index.sort(key=lambda x: x[0])
            
            return [emb for _, emb in embeddings_with_index]
        
        return await self._retry_request(request, "Sparse embedding service")
    
    async def health_check(self) -> dict:
        """
        Check health of both embedding services.
        
        Returns:
            Dictionary with health status of both services
        """
        results = {
            "dense": {"status": "unknown"},
            "sparse": {"status": "unknown"}
        }
        
        # Check dense service
        try:
            response = await self._client.get(f"{self.dense_url}/health", timeout=5.0)
            results["dense"] = response.json()
        except Exception as e:
            results["dense"] = {"status": "error", "error": str(e)}
        
        # Check sparse service
        try:
            response = await self._client.get(f"{self.sparse_url}/health", timeout=5.0)
            results["sparse"] = response.json()
        except Exception as e:
            results["sparse"] = {"status": "error", "error": str(e)}
        
        return results
    
    async def close(self):
        """Close the HTTP client connection pool."""
        await self._client.aclose()


# Global singleton instance
embedding_client = EmbeddingClient()