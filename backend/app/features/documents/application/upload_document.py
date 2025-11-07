"""Document upload and processing pipeline orchestrator."""

import asyncio
from typing import List
import uuid
from datetime import datetime
from hashlib import sha256
from app.features.documents.domain.entities import Document
from app.features.documents.domain.value_objects import FileUpload
from app.features.documents.domain.repository_interface import EmbeddingRepository
from app.features.documents.application.process_document import ProcessDocument
from app.features.documents.application.chunk_document import ChunkDocument
from app.features.documents.application.embed_document import EmbedDocument
from app.features.documents.application.upsert_document import UpsertDocument
from app.features.documents.infrastructure.llm_groq_service import get_llm_service
from app.core.config import settings
from app.core.logger import get_logger
from app.core.supabase_client import get_supabase


logger = get_logger(__name__)


class UploadDocument:

    def __init__(self, embedding_repo: EmbeddingRepository):
        llm_service = get_llm_service()
        self.process = ProcessDocument()
        self.chunk = ChunkDocument(llm_service=llm_service)
        self.embed = EmbedDocument()
        self.upsert = UpsertDocument(embedding_repo)
    
    async def _compute_file_hash(self, file_bytes: bytes) -> str:
        return sha256(file_bytes).hexdigest()

    async def _chunk_and_embed_batch(
        self, 
        page_batch: List,
        doc_id: str,
        semaphore: asyncio.Semaphore
    ):
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
        supabase = get_supabase()
        doc_id = str(uuid.uuid4())
        course_id = course_id or "11111111-1111-1111-1111-111111111111"
        try:
            logger.info(f"Processing {file_upload.filename}")
            file_hash = await self._compute_file_hash(file_upload.content)
            storage_path = f"{user_id}/{course_id}/{doc_id}_{file_upload.filename}"


            # Upload file to Supabase Storage
            logger.info(f"Uploading to Supabase Storage: {storage_path}")
            
            supabase.storage.from_(settings.supabase_bucket_documents).upload(
                path=storage_path,
                file=file_upload.content,
                file_options={"content-type": file_upload.content_type}
            )
            
            logger.info(f"File uploaded successfully to {storage_path}")

            # Process document
            page_documents, total_pages, _ = await self.process.execute(file_upload)
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
                mime_type=file_upload.content_type,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                error_message=None,
            )
            
            # Save initial document metadata to PostgreSQL
            supabase.table("documents").insert({
                "id": doc.id,
                "user_id": user_id,
                "course_id": course_id,
                "filename": file_upload.filename,  # Sanitized filename for storage
                "storage_path": storage_path,
                "file_hash": file_hash,
                "file_size": len(file_upload.content),
                "mime_type": file_upload.content_type,
                "status": "processing",
                "total_pages": total_pages,
                "chunks_count": 0,
            }).execute()
            
            logger.info(f"Document metadata saved to database: {doc.id}")
            
            page_batches = [
                page_documents[i:i + settings.pages_per_batch]
                for i in range(0, len(page_documents), settings.pages_per_batch)
            ]
            
            logger.info(f"Processing {total_pages} pages in {len(page_batches)} batches")
            
            semaphore = asyncio.Semaphore(settings.max_parallel_pages)
            
            tasks = []
            for batch in page_batches:
                task = self._chunk_and_embed_batch(batch, doc.id, semaphore)
                tasks.append(task)
            
            all_results = await asyncio.gather(*tasks)
            
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
            
            indexed_count = 0
            if all_chunks:
                indexed_count = await self.upsert.execute(
                    user_id=user_id,
                    course_id=course_id,
                    chunks=all_chunks,
                    dense_embeddings=all_dense,
                    sparse_embeddings=all_sparse
                )
            
            doc.chunks_count = indexed_count
            doc.mark_as_completed(total_pages=total_pages)
            total_pages = all_results[0][0][-1].page_number if all_results[0][0] else 0
            # Update document status in database
            supabase.table("documents").update({
                "status": "completed",
                "chunks_count": indexed_count,
            }).eq("id", doc.id).execute()
            
            logger.info(f"Completed: {indexed_count} chunks from {total_pages} pages")
            
            return doc

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            
            # Update document status to failed
            try:
                supabase.table("documents").update({
                    "status": "failed",
                    "error_message": str(e),
                }).eq("id", doc_id).execute()
            except Exception as db_error:
                logger.error(f"Failed to update document status: {db_error}")
            
            raise