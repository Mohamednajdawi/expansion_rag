"""Pydantic models for RAG API."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Response for document processing."""
    document_id: str
    filename: str
    size: int
    success: bool = True
    message: Optional[str] = None


class TextDocumentRequest(BaseModel):
    """Request for processing a text document."""
    content: str = Field(..., description="The text content of the document")
    filename: Optional[str] = Field(None, description="Optional filename")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class ChunkResponse(BaseModel):
    """Response model for a retrieved text chunk."""
    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QARequest(BaseModel):
    """Request for question answering."""
    query: str = Field(..., description="The question to answer")
    top_k: Optional[int] = Field(3, description="Number of chunks to retrieve")
    model: Optional[str] = Field("gpt-4o-mini", description="OpenAI model to use for generation")
    temperature: Optional[float] = Field(0.0, description="Sampling temperature")


class QAResponse(BaseModel):
    """Response for question answering."""
    answer: str
    chunks: List[ChunkResponse]
    expanded_queries: Optional[List[str]] = Field(default_factory=list, description="Expanded queries used for retrieval")
    success: bool 