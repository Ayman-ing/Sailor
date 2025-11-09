"""API routes for the documents feature."""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Form
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.documents.application.upload_document import UploadDocument
from app.features.documents.domain.value_objects import FileUpload
from app.features.documents.presentation.schemas import DocumentResponse
from app.features.documents.domain.entities import DEFAULT_USER_ID

# Infrastructure implementations
from app.features.documents.infrastructure.document_repository_pg import DocumentRepositoryPg
from app.features.documents.infrastructure.storage_repository_supabase import SupabaseStorageRepository
from app.features.documents.infrastructure.embedding_repository_qdrant import EmbeddingRepositoryQdrant

# Core dependencies
from app.core.qdrant_client import qdrant_manager
from app.core.database import get_db_session  # You need this for PostgreSQL session


# --- Dependency Injection Setup ---

async def get_upload_use_case(
    session: AsyncSession = Depends(get_db_session)
) -> UploadDocument:
    """Dependency to provide the UploadDocument use case with all repositories.
    
    Args:
        session: Database session from FastAPI dependency
        
    Returns:
        Configured UploadDocument use case with injected dependencies
    """
    # Create repository instances
    document_repo = DocumentRepositoryPg(session)
    storage_repo = SupabaseStorageRepository()
    embedding_repo = EmbeddingRepositoryQdrant(qdrant_manager)
    
    # Inject dependencies into use case
    return UploadDocument(
        document_repo=document_repo,
        storage_repo=storage_repo,
        embedding_repo=embedding_repo
    )

# --- End of Dependency Injection ---


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document_endpoint(
    file: UploadFile = File(...),
    course_id: Optional[str] = Form(None),
    use_case: UploadDocument = Depends(get_upload_use_case)
):
    """
    Uploads a document for processing. The file is processed asynchronously.
    
    Args:
        file: The PDF file to upload
        course_id: Optional course ID to associate the document with
        
    Returns:
        DocumentResponse with processing status and metadata
        
    Raises:
        HTTPException: If file processing fails
    """
    try:
        # Validate file type
        if not file.content_type == "application/pdf":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Read file content
        content = await file.read()
        
        # Validate file size (e.g., max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB in bytes
        if len(content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of 50MB"
            )

        # Create the FileUpload value object
        file_upload = FileUpload(
            filename=file.filename,
            content=content,
            content_type=file.content_type
        )
        
        # Execute the entire pipeline
        # Using a default user ID since we don't have auth yet
        processed_doc = await use_case.execute(DEFAULT_USER_ID, file_upload, course_id)

        return processed_doc

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )