"""Pydantic models for RAG API."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


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


class FileListResponse(BaseModel):
    """Response containing list of files."""
    files: List[str] = Field(..., description="List of file names")
    total_files: int = Field(..., description="Total number of files")


class Message(BaseModel):
    """A chat message."""
    role: str  # "user" or "assistant"
    content: str


class ChunkResponse(BaseModel):
    """Response model for a retrieved text chunk."""
    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """A chat request with optional conversation history."""
    message: str
    history: Optional[List[Message]] = None
    top_k: Optional[int] = 3
    model: Optional[str] = "gpt-4.1-mini-2025-04-14"
    temperature: Optional[float] = 0.0
    meta_information: Optional[str] = None


class ChatResponse(BaseModel):
    """A chat response with additional context."""
    message: Message
    chunks: List[ChunkResponse]
    expanded_queries: List[str]
    success: bool


class QARequest(BaseModel):
    """Request for question answering."""
    query: str = Field(..., description="The question to answer")
    top_k: Optional[int] = Field(3, description="Number of chunks to retrieve")
    model: Optional[str] = Field("gpt-4.1-mini-2025-04-14", description="OpenAI model to use for generation")
    temperature: Optional[float] = Field(0.0, description="Sampling temperature")


class QAResponse(BaseModel):
    """Response for question answering."""
    answer: str
    chunks: List[ChunkResponse]
    expanded_queries: Optional[List[str]] = Field(default_factory=list, description="Expanded queries used for retrieval")
    success: bool 