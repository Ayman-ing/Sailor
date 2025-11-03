"""PostgreSQL implementation of the DocumentRepository interface."""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.features.documents.domain.entities import Document, DocumentMarkdown, DocumentMetadata
from app.features.documents.domain.repository_interface import DocumentRepository
from app.features.documents.infrastructure.models import DocumentModel, DocumentMarkdownModel
from app.shared.exceptions import RepositoryError

# --- Mappers to convert between domain entities and DB models ---

def _to_document_entity(model: DocumentModel) -> Document:
    """Converts a DocumentModel (SQLAlchemy) to a Document (domain entity)."""
    metadata = DocumentMetadata(**model.metadata) if model.metadata else None
    return Document(
        id=model.id,
        user_id=model.user_id,
        filename=model.filename,
        file_hash=model.file_hash,
        file_size_bytes=model.file_size_bytes,
        total_pages=model.total_pages,
        status=model.status,
        metadata=metadata,
        created_at=model.created_at,
        updated_at=model.updated_at,
        processed_at=model.processed_at,
        error_message=model.error_message,
    )

def _to_markdown_entity(model: DocumentMarkdownModel) -> DocumentMarkdown:
    """Converts a DocumentMarkdownModel to a DocumentMarkdown entity."""
    return DocumentMarkdown(
        document_id=model.document_id,
        content=model.content,
        extracted_at=model.extracted_at,
    )

class DocumentRepositoryPg(DocumentRepository):
    """PostgreSQL repository for documents."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, document: Document) -> Document:
        """Saves or updates a document entity in the database."""
        try:
            # Check if the document already exists
            stmt = select(DocumentModel).where(DocumentModel.id == document.id)
            result = await self.session.execute(stmt)
            model = result.scalars().first()

            if not model:
                model = DocumentModel(id=document.id)

            # Update model fields from entity
            model.user_id = document.user_id
            model.filename = document.filename
            model.file_hash = document.file_hash
            model.file_size_bytes = document.file_size_bytes
            model.total_pages = document.total_pages
            model.status = document.status
            model.metadata = document.metadata.to_dict() if document.metadata else None
            model.created_at = document.created_at
            model.updated_at = document.updated_at
            model.processed_at = document.processed_at
            model.error_message = document.error_message

            self.session.add(model)
            await self.session.flush()
            return _to_document_entity(model)
        except Exception as e:
            raise RepositoryError(f"Error saving document: {e}")

    async def find_by_id(self, document_id: str) -> Optional[Document]:
        """Finds a document by its ID."""
        stmt = select(DocumentModel).where(DocumentModel.id == document_id)
        result = await self.session.execute(stmt)
        model = result.scalars().first()
        return _to_document_entity(model) if model else None

    async def find_by_hash(self, user_id: str, file_hash: str) -> Optional[Document]:
        """Finds a document by its file hash for a specific user."""
        stmt = select(DocumentModel).where(
            DocumentModel.user_id == user_id,
            DocumentModel.file_hash == file_hash
        )
        result = await self.session.execute(stmt)
        model = result.scalars().first()
        return _to_document_entity(model) if model else None

    async def save_markdown(self, markdown: DocumentMarkdown) -> None:
        """Saves the extracted markdown content."""
        try:
            model = DocumentMarkdownModel(
                id=markdown.document_id, # Use document_id as primary key
                document_id=markdown.document_id,
                content=markdown.content,
                extracted_at=markdown.extracted_at
            )
            await self.session.merge(model)
            await self.session.flush()
        except Exception as e:
            raise RepositoryError(f"Error saving markdown: {e}")