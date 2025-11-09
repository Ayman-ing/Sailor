"""Document upload and processing pipeline orchestrator."""

import asyncio
from typing import List
import uuid
from datetime import datetime
from hashlib import sha256
from app.features.documents.domain.entities import Document
from app.features.documents.domain.value_objects import FileUpload
from app.features.documents.domain.repository_interface import (
    DocumentRepository, 
    EmbeddingRepository, 
    StorageRepository
)
from app.features.documents.application.process_document import ProcessDocument
from app.features.documents.application.chunk_document import ChunkDocument
from app.features.documents.application.embed_document import EmbedDocument
from app.features.documents.application.upsert_document import UpsertDocument
from app.features.documents.infrastructure.llm_groq_service import get_llm_service
from app.core.config import settings
from app.core.logger import get_logger


logger = get_logger(__name__)


class UploadDocument:
    """Orchestrates the document upload and processing pipeline."""

    def __init__(
        self, 
        document_repo: DocumentRepository,
        storage_repo: StorageRepository,
        embedding_repo: EmbeddingRepository
    ):
        """Initialize the upload document use case.
        
        Args:
            document_repo: Repository for document metadata persistence
            storage_repo: Repository for file storage operations
            embedding_repo: Repository for vector embeddings
        """
        llm_service = get_llm_service()
        self.process = ProcessDocument()
        self.chunk = ChunkDocument(llm_service=llm_service)
        self.embed = EmbedDocument()
        self.upsert = UpsertDocument(embedding_repo)
        self.storage_repo = storage_repo
        self.document_repo = document_repo
    
    async def _compute_file_hash(self, file_bytes: bytes) -> str:
        """Compute SHA-256 hash of file content."""
        return sha256(file_bytes).hexdigest()

    async def _chunk_and_embed_batch(
        self, 
        page_batch: List,
        doc_id: str,
        semaphore: asyncio.Semaphore
    ):
        """Process a batch of pages: chunk and embed them.
        
        Args:
            page_batch: List of page tuples to process
            doc_id: Document ID
            semaphore: Semaphore for concurrency control
            
        Returns:
            Tuple of (chunks, dense_embeddings, sparse_embeddings)
        """
        async with semaphore:
            batch_start = page_batch[0][1]
            batch_end = page_batch[-1][1]
            
            try:
                all_batch_chunks = []
                for page_tuple in page_batch:
                    page_chunks = await self.chunk.execute([page_tuple], doc_id)
                    all_batch_chunks.extend(page_chunks)
                
                if not all_batch_chunks:
                    return ([], [], [])
                
                dense_emb, sparse_emb = await self.embed.execute(all_batch_chunks)
                
                logger.info(f"Batch {batch_start}-{batch_end}: {len(all_batch_chunks)} chunks embedded")
                return (all_batch_chunks, dense_emb, sparse_emb)
                
            except Exception as e:
                logger.error(f"Batch {batch_start}-{batch_end} FAILED - {len(page_batch)} pages SKIPPED: {e}")
                return ([], [], [])

    async def execute(self, user_id: str, file_upload: FileUpload, course_id: str | None = None) -> Document:
        """Execute the document upload and processing pipeline.
        
        Args:
            user_id: ID of the user uploading the document
            file_upload: FileUpload object containing file data
            course_id: Optional course ID to associate with document
            
        Returns:
            Processed Document entity
            
        Raises:
            Exception: If document already exists or processing fails
        """
        doc_id = str(uuid.uuid4())
        course_id = course_id or "11111111-1111-1111-1111-111111111111"
        
        # Initialize document object for error handling
        doc = None
        
        try:
            logger.info(f"Processing {file_upload.filename}")
            
            # Check if document with same hash already exists for user
            file_hash = await self._compute_file_hash(file_upload.content)
            existing_doc = await self.document_repo.get_by_hash(user_id, file_hash)
            if existing_doc:
                raise Exception(f"Document with the same content already exists (ID: {existing_doc.id})")

            # Build storage path
            storage_path = f"{user_id}/{course_id}/{doc_id}_{file_upload.filename}"
            
            # Upload file to Supabase Storage
            logger.info(f"Uploading to Supabase Storage: {storage_path}")
            await self.storage_repo.upload_file(storage_path, file_upload)
            logger.info(f"File uploaded successfully to {storage_path}")

            # Process document (extract pages)
            page_documents, total_pages, _ = await self.process.execute(file_upload)

            # Save initial document metadata to PostgreSQL
            doc = Document(
                id=doc_id,
                user_id=user_id,
                course_id=course_id,
                filename=file_upload.filename,
                storage_path=storage_path,
                file_hash=file_hash,
                file_size=len(file_upload.content),
                total_pages=total_pages,
                chunks_count=0,
                status="processing",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                error_message=None,
            )
            await self.document_repo.save(doc)
            logger.info(f"Document metadata saved to database: {doc.id}")
            
            # Split pages into batches
            page_batches = [
                page_documents[i:i + settings.pages_per_batch]
                for i in range(0, len(page_documents), settings.pages_per_batch)
            ]
            
            logger.info(f"Processing {total_pages} pages in {len(page_batches)} batches")
            
            # Semaphore for controlling parallelism
            semaphore = asyncio.Semaphore(settings.max_parallel_pages)
            
            # Create tasks for parallel processing
            tasks = []
            for batch in page_batches:
                task = self._chunk_and_embed_batch(batch, doc.id, semaphore)
                tasks.append(task)
            
            # Wait for all batches to complete
            all_results = await asyncio.gather(*tasks)
            
            # Collect results
            all_chunks = []
            all_dense = []
            all_sparse = []
            failed_batches = 0
            
            for chunks, dense, sparse in all_results:
                if not chunks:
                    failed_batches += 1
                all_chunks.extend(chunks)
                all_dense.extend(dense)
                all_sparse.extend(sparse)
            
            if failed_batches > 0:
                logger.warning(f"{failed_batches} batch(es) failed - some pages were skipped")
            
            # Upsert embeddings to vector database
            indexed_count = 0
            if all_chunks:
                indexed_count = await self.upsert.execute(
                    user_id=user_id,
                    course_id=course_id,
                    chunks=all_chunks,
                    dense_embeddings=all_dense,
                    sparse_embeddings=all_sparse
                )
            
            # Update document with final counts and status
            doc.chunks_count = indexed_count
            doc.status = "completed"
            doc.updated_at = datetime.utcnow()
            
            await self.document_repo.update(doc)
            
            logger.info(f"Completed: {indexed_count} chunks from {total_pages} pages")
            
            return doc

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            
            # Update document status to failed if doc was created
            if doc:
                try:
                    doc.status = "failed"
                    doc.error_message = str(e)
                    doc.updated_at = datetime.utcnow()
                    await self.document_repo.update(doc)
                except Exception as db_error:
                    logger.error(f"Failed to update document status: {db_error}")
            
            raise