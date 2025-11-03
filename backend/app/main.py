from fastapi import FastAPI
#from contextlib import asynccontextmanager

#from app.core.database import db_manager
from app.core.logger import get_logger
from app.api_router import api_router # <-- Import the central router

logger = get_logger(__name__)

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Handles application startup and shutdown events."""
#     logger.info("Application startup...")
#     # You can uncomment this if you want to create DB tables on startup
#     # await db_manager.create_tables()
#     yield
#     logger.info("Application shutdown...")
#     await db_manager.close()

app = FastAPI(
    title="Sailor API",
    description="AI-powered student assistant",
    version="0.1.0",
    
)

# Include the single, aggregated API router with a global prefix
app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["Health"])
async def read_root():
    """Root endpoint for health checks."""
    return {"status": "ok"}
