"""Orchestrator use case for the entire document processing pipeline."""

from app.features.documents.domain.entities import Document
from app.features.documents.domain.value_objects import FileUpload
from app.features.documents.domain.repository_interface import EmbeddingRepository
from app.features.documents.application.process_document import ProcessDocument
from app.features.documents.application.chunk_document import ChunkDocument
from app.features.documents.application.index_document import IndexDocument
from app.features.documents.infrastructure.llm_groq_service import get_llm_service
from app.shared.helpers import generate_file_hash
from app.core.logger import get_logger

logger = get_logger(__name__)


class UploadDocument:
    """Orchestrates the document upload, processing, chunking, and indexing pipeline."""

    def __init__(
        self,
        #doc_repo: DocumentRepository,
        embedding_repo: EmbeddingRepository,
    ):
        #self.doc_repo = doc_repo
        llm_service = get_llm_service()
        self.process = ProcessDocument()
        self.chunk = ChunkDocument(llm_service=llm_service)
        self.index = IndexDocument(embedding_repo)

    async def execute(self, user_id: str, file_upload: FileUpload) -> Document:
        """
        Executes the full document processing pipeline.
        
        1. Checks for duplicates.
        2. Creates a Document record in the database.
        3. Processes, chunks, and indexes the document.
        4. Updates the document status (completed or failed).
        
        Returns:
            The final Document entity with its status.
        """
        file_hash = generate_file_hash(file_upload.content)
        
        # 1. Check for duplicates for this user
        # existing_doc = await self.doc_repo.find_by_hash(user_id, file_hash)
        # if existing_doc:
        #     logger.info(f"Document '{file_upload.filename}' with hash {file_hash} already exists for user {user_id}.")
        #     return existing_doc

        # 2. Create and save the initial document record
        doc = Document(
            user_id=user_id,
            filename=file_upload.filename,
            file_hash=file_hash,
            file_size_bytes=len(file_upload.content),
            status="pending"
        )
        #await self.doc_repo.save(doc)
        
        try:
            # --- Start Pipeline ---
            logger.info(f"Starting pipeline for document ID: {doc.id}")
            doc.mark_as_processing()

            # Step 1: Process PDF -> List of (MarkdownDocument, page_number) per page
            page_documents, total_pages, page_chunks = await self.process.execute(file_upload)
            
            # Step 2: Chunk all pages -> List[DocumentChunk] with accurate page numbers
            chunks = await self.chunk.execute(page_documents, doc.id)
            
            # Step 3: Index chunks -> Qdrant
            indexed_count = await self.index.execute(user_id, doc.id, chunks)
            
            # --- Finalize Success ---
            doc.mark_as_completed(total_pages=total_pages)
            doc.chunk_count = indexed_count
            #await self.doc_repo.save(doc)
            
            logger.info(f"Pipeline completed successfully for document ID: {doc.id}")
            
        except Exception as e:
            # --- Handle Failure ---
            logger.error(f"Pipeline failed for document {doc.id}: {e}", exc_info=True)
            doc.mark_as_failed(str(e))
            #await self.doc_repo.save(doc)
        
        return doc