"""Document handling routes."""
import os
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..models import DocumentResponse, TextDocumentRequest
from ..core.document_processor import process_text_document, save_uploaded_file, get_document_content
from ..core.embeddings import create_document_embeddings, verify_document_embeddings, process_missing_embeddings

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/embedding-status")
async def get_embedding_status():
    """Get the status of document embeddings."""
    return verify_document_embeddings()


@router.post("/process-missing-embeddings")
async def process_missing():
    """Process embeddings for any documents that are missing them."""
    return process_missing_embeddings()


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document file and process it."""
    try:
        document_info = save_uploaded_file(file.file, file.filename)
        
        # Process embeddings
        content = document_info.get("content", "")
        if content:
            create_document_embeddings(
                document_info["document_id"],
                content,
                document_info["metadata"]
            )
        
        return DocumentResponse(
            document_id=document_info["document_id"],
            filename=document_info["filename"],
            size=document_info["size"],
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post("/text", response_model=DocumentResponse)
async def process_text(request: TextDocumentRequest):
    """Process a text document directly."""
    try:
        document_info = process_text_document(
            request.content,
            request.filename,
            request.metadata
        )
        
        # Process embeddings
        create_document_embeddings(
            document_info["document_id"],
            request.content,
            document_info["metadata"]
        )
        
        return DocumentResponse(
            document_id=document_info["document_id"],
            filename=document_info["filename"],
            size=document_info["size"],
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get document information."""
    content = get_document_content(document_id)
    if not content:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        document_id=document_id,
        filename=f"{document_id}.txt",  # Simplified for now
        size=len(content),
        success=True
    ) 