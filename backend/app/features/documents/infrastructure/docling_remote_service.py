"""
Client for remote Docling processing service.
Use this when you want to offload heavy PDF processing to a more powerful machine.
"""
import httpx
from typing import List
from app.features.documents.domain.entities import DocumentChunk
from app.features.documents.domain.value_objects import FileUpload
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class RemoteDoclingService:
    """
    Client for a remote Docling processing service.
    Sends PDFs to a separate service for processing and receives chunks.
    """
    
    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip('/')
        self.timeout = httpx.Timeout(300.0)  # 5 minutes for heavy processing
    
    async def process_and_chunk(
        self, 
        file_upload: FileUpload, 
        document_id: str
    ) -> List[DocumentChunk]:
        """
        Send PDF to remote service and get chunks back.
        """
        logger.info(f"Sending document {document_id} to remote Docling service at {self.service_url}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Prepare file for upload
                files = {
                    'file': (file_upload.filename, file_upload.content, 'application/pdf')
                }
                
                # Send to remote service
                response = await client.post(
                    f"{self.service_url}/process",
                    files=files
                )
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                logger.info(f"Received {data['total_chunks']} chunks from remote service")
                
                # Convert to domain entities
                chunks = []
                for chunk_data in data['chunks']:
                    chunks.append(
                        DocumentChunk(
                            document_id=document_id,
                            content=chunk_data['text'],
                            chunk_index=chunk_data['chunk_index'],
                            token_count=chunk_data['token_count'],
                            metadata=chunk_data['metadata']
                        )
                    )
                
                return chunks
                
        except httpx.HTTPError as e:
            logger.error(f"Remote Docling service request failed: {e}")
            raise Exception(f"Failed to process document via remote service: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with remote Docling service: {e}", exc_info=True)
            raise


def get_remote_docling_service() -> RemoteDoclingService:
    """Factory function to create RemoteDoclingService."""
    if not settings.docling_service_url:
        raise ValueError("docling_service_url not configured in settings")
    return RemoteDoclingService(settings.docling_service_url)
