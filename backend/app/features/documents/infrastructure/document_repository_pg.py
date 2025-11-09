"""PostgreSQL implementation of the DocumentRepository interface."""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.features.documents.domain.entities import Document
from app.features.documents.domain.repository_interface import DocumentRepository
from app.features.documents.infrastructure.models import DocumentModel
from app.shared.exceptions import RepositoryError
from app.core.logger import get_logger
from uuid import UUID


logger = get_logger(__name__)

# --- Mappers to convert between domain entities and DB models ---

def _to_document_entity(model: DocumentModel) -> Document:
    """Converts a DocumentModel (SQLAlchemy) to a Document (domain entity)."""
    return Document(
        id=model.id,
        user_id=model.user_id,
        course_id=model.course_id,
        filename=model.filename,
        file_hash=model.file_hash,
        file_size=model.file_size,
        total_pages=model.total_pages,
        chunks_count=model.chunks_count,
        storage_path=model.storage_path,
        status=model.status,
        created_at=model.created_at,
        updated_at=model.updated_at,
        error_message=model.error_message,
    )

def _to_document_model(entity: Document) -> DocumentModel:
    """Converts a Document (domain entity) to a DocumentModel (SQLAlchemy)."""
    return DocumentModel(
        id=entity.id,
        user_id=entity.user_id,
        course_id=entity.course_id,
        filename=entity.filename,
        file_hash=entity.file_hash,
        file_size=entity.file_size,
        total_pages=entity.total_pages,
        chunks_count=entity.chunks_count,
        storage_path=entity.storage_path,
        status=entity.status,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        error_message=entity.error_message,
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

            if model:
                raise RepositoryError(f"Document with ID {document.id} already exists.")

            model = _to_document_model(document)
            self.session.add(model)
            await self.session.flush()
            return _to_document_entity(model)
        except Exception as e:
            raise RepositoryError(f"Error saving document: {e}")
    async def delete(self, document_id: str) -> None:
        """Deletes a document by its ID."""
        try:
            stmt = select(DocumentModel).where(DocumentModel.id == document_id)
            result = self.session.execute(stmt)
            model = await result.scalars().first()
            if model:
                self.session.delete(model)
                await self.session.flush()
        except Exception as e:
            raise RepositoryError(f"Error deleting document: {e}")
        
    async def update(self, document: Document) -> Document:
        """Updates the filename of an existing document."""
        try:
            stmt = select(DocumentModel).where(DocumentModel.id == document.id)
            result = await self.session.execute(stmt)
            model = result.scalars().first()

            if not model:
                raise RepositoryError(f"Document with ID {document.id} not found for update.")

            # Update the modified fields
            model.filename = document.filename
            model.status = document.status
            model.chunks_count = document.chunks_count
            model.error_message = document.error_message
            
            
            await self.session.flush()
            return _to_document_entity(model)
        except Exception as e:
            raise RepositoryError(f"Error updating document: {e}")
    async def get_by_id(self, document_id: str) -> Optional[Document]:
        """Finds a document by its ID."""
        stmt = select(DocumentModel).where(DocumentModel.id == document_id)
        result = await self.session.execute(stmt)
        model = result.scalars().first()
        return _to_document_entity(model) if model else None

    async def get_by_hash(self, user_id: str, file_hash: str) -> Optional[Document]:

        user_uuid = UUID(user_id)

        """Finds a document by its file hash for a specific user."""
        logger.info(f"Getting document by hash for user {user_id}")
        stmt = select(DocumentModel).where(
               and_(
            DocumentModel.user_id == user_uuid,
            DocumentModel.file_hash == file_hash
        )
        )
        result = await self.session.execute(stmt)
        model = result.scalars().first()
        if model:
            logger.info(f"Document found for user {user_id} with hash {file_hash}")
            return _to_document_entity(model)
        logger.info(f"No document found for user {user_id} with hash {file_hash}")
        return None

    async def get_all_by_user(self, user_id: str) -> List[Document]:
        """Lists all documents for a specific user."""
        user_uuid = UUID(user_id)

        stmt = select(DocumentModel).where(DocumentModel.user_id == user_id)
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [_to_document_entity(model) for model in models]
    
    
    async def get_all_by_user_and_course(self, user_id: str, course_id: str) -> List[Document]:
        """Lists all documents for a specific user and course."""
        user_uuid = UUID(user_id)
        course_uuid = UUID(course_id)
        stmt = select(DocumentModel).where(
            DocumentModel.user_id == user_uuid,
            DocumentModel.course_id == course_uuid
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [_to_document_entity(model) for model in models]
    