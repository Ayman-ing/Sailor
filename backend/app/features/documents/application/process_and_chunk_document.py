"""
A use case to convert and chunk a document using the Docling library.
This replaces the separate process and chunk steps.
"""
import os
import io
import tempfile
from typing import List
from pathlib import Path

from docling.chunking import HybridChunker

from app.features.documents.domain.entities import DocumentChunk
from app.features.documents.domain.value_objects import FileUpload
from app.core.logger import get_logger
from app.core.model_manager import get_model_manager

logger = get_logger(__name__)

class ProcessAndChunkDocument:
    """
    Uses Docling to perform document conversion and hybrid chunking in one step.
    Uses the preloaded Docling converter from the model manager.
    """
    def __init__(self):
        # Get the preloaded converter from the model manager
        self.model_manager = get_model_manager()
        self.converter = self.model_manager.docling_converter
        self.chunker = HybridChunker(chunk_size=512, chunk_overlap=50)

    async def execute(self, file_upload: FileUpload, document_id: str) -> List[DocumentChunk]:
        """
        Converts and chunks an uploaded file by saving it to a temporary file first.
        """
        logger.info(f"Starting Docling processing for document: {document_id}")
        
        temp_file_path = None
        try:
            # Get the original filename and extension
            original_filename = file_upload.filename
            file_extension = Path(original_filename).suffix or ".pdf"
            
            # Create a temporary file with the original filename (sanitized) and extension
            # Use a temporary directory to ensure cleanup
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, original_filename)
            
            logger.info(f"Created temporary file for processing: {temp_file_path}")
            
            # Write the uploaded content to the temporary file
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_upload.content)
            # 1. Convert the document using Docling from the temp file path
            conversion_result = self.converter.convert(temp_file_path)
            
            # 2. Chunk the converted document
            docling_chunks = self.chunker.chunk(conversion_result.document)

            # 3. Map Docling chunks to our domain entity
            domain_chunks = []
            for i, chunk in enumerate(docling_chunks):
                domain_chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        content=chunk.text,
                        chunk_index=i,
                        # Docling's chunker doesn't provide a token count directly, so we approximate
                        token_count=len(chunk.text.split()), 
                        metadata=chunk.meta.export_json_dict() if chunk.meta else {}
                    )
                )
            
            logger.info(f"Docling created {len(domain_chunks)} chunks for document {document_id}")
            return domain_chunks

        except Exception as e:
            logger.error(f"Docling processing failed for document {document_id}: {e}", exc_info=True)
            # Re-raise as a domain-specific exception if needed
            raise
        finally:
            # Ensure the temporary file is always cleaned up
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
