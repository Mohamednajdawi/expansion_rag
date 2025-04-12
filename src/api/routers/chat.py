"""Chat routes for RAG system."""
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from ..models import Message, ChatRequest, ChatResponse
from ..core.rag import generate_answer

router = APIRouter(prefix="/chat", tags=["chat"])

def format_conversation_history(messages: List[Message]) -> str:
    """Format conversation history into a string, including only user questions and assistant responses."""
    formatted_history = ""
    for msg in messages:
        if msg.role in ["user", "assistant"]:
            role = "Assistant" if msg.role == "assistant" else "User"
            formatted_history += f"{role}: {msg.content}\n"
    return formatted_history.strip()

@router.post("/process", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """Process a chat message with conversation history."""
    try:
        # Format conversation history if available
        conversation_history = None
        print("Received history:", request.history)  # Debug: Print received history
        if request.history and len(request.history) > 0:
            conversation_history = format_conversation_history(request.history)
            print("Formatted history:", conversation_history)  # Debug: Print formatted history
        
        # Generate response using RAG
        response = generate_answer(
            query=request.message,
            conversation_history=conversation_history,
            top_k=request.top_k,
            model=request.model,
            temperature=request.temperature,
            meta_information=request.meta_information
        )
        
        # Create the assistant message
        assistant_message = Message(
            role="assistant",
            content=response["answer"]
        )
        
        # No need to convert chunks - use them directly
        return ChatResponse(
            message=assistant_message,
            chunks=response["chunks"],  # Use the full chunk objects
            expanded_queries=response["expanded_queries"],
            success=response["success"]
        )
        
    except Exception as e:
        print(f"Error in process_chat: {e}")  # Add this to see the actual error
        raise HTTPException(status_code=500, detail=str(e)) 