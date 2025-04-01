"""Question answering routes using RAG."""
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from ..models import QARequest, QAResponse, ChunkResponse
from ..core.rag import generate_answer
from ..core.embeddings import verify_document_embeddings, process_missing_embeddings

router = APIRouter(prefix="/qa", tags=["question-answering"])


@router.post("", response_model=QAResponse)
async def answer_question(request: QARequest):
    """
    Answer a question using RAG from all available documents.
    
    This endpoint:
    1. Verifies that all documents have embeddings
    2. If any documents are missing embeddings, processes them automatically
    3. Takes a question
    4. Retrieves relevant chunks from all documents using FAISS similarity search
    5. Generates an answer using OpenAI
    """
    try:
        # Verify document embeddings and process any missing ones
        verification = verify_document_embeddings()
        if not verification["is_complete"]:
            # Process missing embeddings
            processing_result = process_missing_embeddings()
            
            # Check if processing was successful
            if not processing_result["verification"]["is_complete"]:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Failed to process all missing embeddings",
                        "processing_result": processing_result
                    }
                )
        
        # Generate answer using RAG
        result = generate_answer(
            query=request.query,
            top_k=request.top_k or 3,
            model=request.model,
            temperature=request.temperature or 0.0
        )
        
        # Convert chunks to ChunkResponse model
        chunks = [
            ChunkResponse(
                document_id=chunk["document_id"],
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                score=chunk["score"],
                metadata=chunk.get("metadata", {})
            )
            for chunk in result.get("chunks", [])
        ]
        
        return QAResponse(
            answer=result["answer"],
            chunks=chunks,
            expanded_queries=result["expanded_queries"],
            success=result["success"]
        )
    
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}") 