"""API routes for the documents feature."""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status

from app.features.documents.application.upload_document import UploadDocument
from app.features.documents.domain.value_objects import FileUpload
from app.features.documents.presentation.schemas import DocumentResponse
from app.features.documents.domain.entities import DEFAULT_USER_ID

# --- Dependency Injection Setup (Temporary) ---
# In a real app, this would be more robust, likely in a separate dependencies.py
from app.features.documents.infrastructure.embedding_repository_qdrant import EmbeddingRepositoryQdrant
from app.core.qdrant_client import qdrant_manager

def get_upload_use_case() -> UploadDocument:
    """Dependency to provide the UploadDocument use case."""
    embedding_repo = EmbeddingRepositoryQdrant(qdrant_manager)
    return UploadDocument(embedding_repo=embedding_repo)
# --- End of Dependency Injection ---


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document_endpoint(
    file: UploadFile = File(...),
    use_case: UploadDocument = Depends(get_upload_use_case)
):
    """
    Uploads a document for processing. The file is processed asynchronously.
    """
    try:
        # Read file content
        content = await file.read()

        # Create the FileUpload value object
        file_upload = FileUpload(
            filename=file.filename,
            content=content,
            content_type=file.content_type
        )
        
        # Execute the entire pipeline
        # Using a default user ID since we don't have auth yet
        processed_doc = await use_case.execute(DEFAULT_USER_ID, file_upload)

        return processed_doc

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )