"""API routes for the chat feature."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.features.chat.application.query_documents import QueryDocuments
from app.features.chat.application.chat_with_context import ChatWithContext
from app.features.chat.infrastructure.retriever_qdrant import RetrieverQdrant
from app.features.chat.presentation.schemas import (
    ChatRequest,
    ChatResponse as ChatResponseSchema,
    QueryRequest,
    QueryResponse,
    SourceSchema
)
from app.features.chat.domain.entities import RetrievedChunk
from app.features.documents.domain.entities import DEFAULT_USER_ID
from app.features.documents.infrastructure.llm_groq_service import get_llm_service
from app.core.qdrant_client import qdrant_manager
from app.core.logger import get_logger

logger = get_logger(__name__)


# --- Dependency Injection ---
def get_retriever() -> RetrieverQdrant:
    """Dependency to provide the Qdrant retriever."""
    return RetrieverQdrant(qdrant_manager)


def get_query_use_case(retriever: RetrieverQdrant = Depends(get_retriever)) -> QueryDocuments:
    """Dependency to provide the QueryDocuments use case."""
    return QueryDocuments(retriever)


def get_chat_use_case(
    query_use_case: QueryDocuments = Depends(get_query_use_case)
) -> ChatWithContext:
    """Dependency to provide the ChatWithContext use case."""
    llm_service = get_llm_service()
    return ChatWithContext(query_use_case, llm_service)


# --- Router Setup ---
router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


def _chunk_to_source_schema(chunk: RetrievedChunk) -> SourceSchema:
    """Convert RetrievedChunk to SourceSchema."""
    return SourceSchema(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        content=chunk.content,
        score=chunk.score,
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        source_info=chunk.get_source_info()
    )


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_documents_endpoint(
    request: QueryRequest,
    query_use_case: QueryDocuments = Depends(get_query_use_case)
):
    """
    Query documents and retrieve relevant chunks without generating an answer.
    Useful for inspecting what context would be used.
    """
    try:
        from app.features.chat.domain.entities import QueryContext
        from datetime import datetime
        
        # Create query context
        context = QueryContext(
            query=request.query,
            user_id=DEFAULT_USER_ID,
            document_ids=request.document_ids,
            top_k=request.top_k,
            hybrid_alpha=request.hybrid_alpha,
            expand_context=request.expand_context,
            score_threshold=request.score_threshold
        )
        
        # Retrieve chunks
        chunks = await query_use_case.execute(context)
        
        # Convert to response schema
        sources = [_chunk_to_source_schema(chunk) for chunk in chunks]
        
        return QueryResponse(
            query=request.query,
            sources=sources,
            created_at=datetime.utcnow(),
            metadata={
                "top_k": request.top_k,
                "hybrid_alpha": request.hybrid_alpha,
                "expand_context": request.expand_context,
                "score_threshold": request.score_threshold,
                "num_sources": len(sources)
            }
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@router.post("/ask", response_model=ChatResponseSchema, status_code=status.HTTP_200_OK)
async def chat_endpoint(
    request: ChatRequest,
    chat_use_case: ChatWithContext = Depends(get_chat_use_case)
):
    """
    Ask a question and get an AI-generated answer based on your documents.
    Uses RAG (Retrieval-Augmented Generation) to provide context-aware answers.
    """
    try:
        # Execute RAG pipeline
        response = await chat_use_case.execute(
            user_id=DEFAULT_USER_ID,
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k,
            hybrid_alpha=request.hybrid_alpha,
            expand_context=request.expand_context,
            score_threshold=request.score_threshold
        )
        
        # Convert to response schema
        sources = [_chunk_to_source_schema(chunk) for chunk in response.sources]
        
        return ChatResponseSchema(
            id=response.id,
            query=response.query,
            answer=response.answer,
            sources=sources,
            model=response.model,
            created_at=response.created_at,
            metadata=response.metadata
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )
