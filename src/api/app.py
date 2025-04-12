"""FastAPI application for RAG API."""
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routers import documents, qa, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create necessary directories
    os.makedirs(os.getenv("DOCUMENTS_DIR", "./data/documents"), exist_ok=True)
    os.makedirs(os.getenv("EMBEDDINGS_DIR", "./data/embeddings"), exist_ok=True)
    yield
    # Shutdown: Nothing to clean up for now


# Create FastAPI app
app = FastAPI(
    title="RAG API",
    description="A simple Retrieval Augmented Generation API using OpenAI and FAISS",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(qa.router)
app.include_router(chat.router)


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


def start():
    """Run the API using uvicorn."""
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    start() 