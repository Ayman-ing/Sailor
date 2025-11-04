"""
Main API router that aggregates all feature-specific routers.
"""

from fastapi import APIRouter

from app.features.documents.presentation.routes import router as documents_router
from app.features.chat.presentation.routes import router as chat_router

# This is the main router that will be included in the FastAPI app instance.
api_router = APIRouter()

# Include the documents router with its own prefix.
api_router.include_router(documents_router, prefix="/documents", tags=["Documents"])

# Include the chat router with its own prefix.
api_router.include_router(chat_router, prefix="/chat", tags=["Chat"])