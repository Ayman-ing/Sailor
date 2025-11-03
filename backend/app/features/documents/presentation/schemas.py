"""Pydantic schemas (DTOs) for the documents API."""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DocumentResponse(BaseModel):
    """Response model for a document processing operation."""
    id: str
    filename: str
    status: str
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True