"""
Main API router that aggregates all feature-specific routers.
"""

from fastapi import APIRouter

from app.features.documents.presentation.routes import router as documents_router
# from app.features.chat.presentation.routes import router as chat_router # Future
# from app.features.users.presentation.routes import router as users_router # Future

# This is the main router that will be included in the FastAPI app instance.
api_router = APIRouter()

# Include the documents router with its own prefix.
api_router.include_router(documents_router, prefix="/documents", tags=["Documents"])

# When you add more features, you would include their routers here:
# api_router.include_router(chat_router, prefix="/chat", tags=["Chat"])
# api_router.include_router(users_router, prefix="/users", tags=["Users"])