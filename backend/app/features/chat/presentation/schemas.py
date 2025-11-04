"""Pydantic schemas (DTOs) for the chat API."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SourceSchema(BaseModel):
    """Schema for a source chunk in the response."""
    
    chunk_id: str
    document_id: str
    content: str
    score: float
    page_number: int = 0
    chunk_index: int = 0
    source_info: str = Field(default="", description="Formatted source citation")
    
    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    
    query: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    document_ids: Optional[List[str]] = Field(None, description="Optional list of document IDs to search in")
    top_k: int = Field(5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    hybrid_alpha: float = Field(0.7, ge=0.0, le=1.0, description="Balance between dense and sparse search (0=sparse, 1=dense)")
    expand_context: bool = Field(True, description="Whether to retrieve neighboring chunks for high-scoring results")
    score_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum score to trigger context expansion")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the key concepts in chapter 3?",
                "document_ids": None,
                "top_k": 5,
                "hybrid_alpha": 0.7,
                "expand_context": True,
                "score_threshold": 0.7
            }
        }


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    
    id: str
    query: str
    answer: str
    sources: List[SourceSchema]
    model: str
    created_at: datetime
    metadata: dict = Field(default_factory=dict)
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "What are the key concepts?",
                "answer": "Based on the documents, the key concepts are...",
                "sources": [
                    {
                        "chunk_id": "abc123",
                        "document_id": "doc456",
                        "content": "The key concepts include...",
                        "score": 0.95,
                        "page_number": 5,
                        "chunk_index": 0,
                        "source_info": "Document Title (Page 5)"
                    }
                ],
                "model": "llama3-70b-8192",
                "created_at": "2024-01-01T12:00:00Z",
                "metadata": {
                    "top_k": 5,
                    "num_sources": 1
                }
            }
        }


class QueryRequest(BaseModel):
    """Request schema for query-only endpoint (no LLM generation)."""
    
    query: str = Field(..., min_length=1, max_length=2000)
    document_ids: Optional[List[str]] = None
    top_k: int = Field(5, ge=1, le=20)
    hybrid_alpha: float = Field(0.7, ge=0.0, le=1.0)
    expand_context: bool = Field(True, description="Whether to retrieve neighboring chunks for high-scoring results")
    score_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum score to trigger context expansion")


class QueryResponse(BaseModel):
    """Response schema for query-only endpoint."""
    
    query: str
    sources: List[SourceSchema]
    created_at: datetime
    metadata: dict = Field(default_factory=dict)
    
    class Config:
        from_attributes = True
