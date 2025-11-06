"""Document upload and processing pipeline orchestrator."""

import asyncio
from typing import List
import uuid
from datetime import datetime

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

logger = get_logger(__name__)


class UploadDocument:

    def __init__(self, embedding_repo: EmbeddingRepository):
        llm_service = get_llm_service()
        self.process = ProcessDocument()
        self.chunk = ChunkDocument(llm_service=llm_service)
        self.embed = EmbedDocument()
        self.upsert = UpsertDocument(embedding_repo)
    
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

    async def execute(self, user_id: str, file_upload: FileUpload) -> Document:
        try:
            logger.info(f"Processing {file_upload.filename}")
            page_documents, total_pages, _ = await self.process.execute(file_upload)
            
            doc = Document(
                id=str(uuid.uuid4()),
                user_id=user_id,
                filename=file_upload.filename,
                file_hash="",
                file_size_bytes=len(file_upload.content),
                total_pages=total_pages,
                chunk_count=0,
                status="processing",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            
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
                for idx, chunk in enumerate(all_chunks):
                    chunk.chunk_index = idx
                
                indexed_count = await self.upsert.execute(
                    user_id=user_id,
                    document_id=doc.id,
                    chunks=all_chunks,
                    dense_embeddings=all_dense,
                    sparse_embeddings=all_sparse
                )
            
            doc.chunk_count = indexed_count
            doc.mark_as_completed(total_pages=total_pages)
            
            logger.info(f"Completed: {indexed_count} chunks from {total_pages} pages")
            
            return doc

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise