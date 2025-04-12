"""Document embedding using OpenAI API."""
import os
from typing import Dict, List, Optional, Any
import numpy as np
import tiktoken
import openai
from openai import OpenAI
import faiss
import pickle
import json
from pathlib import Path
from dotenv import load_dotenv
from ..core.document_processor import get_document_content

# Load environment variables
load_dotenv()

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Default embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
# Default encoding for token counting
ENCODING = tiktoken.get_encoding("cl100k_base")
# Maximum tokens for embedding model
MAX_TOKENS = 8191
# Path to store the FAISS index
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", "./data/embeddings"))

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Get embeddings for a text using OpenAI API."""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks of tokens."""
    tokens = ENCODING.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        if len(chunk_tokens) < 10:  # Skip very small chunks
            continue
        chunks.append(ENCODING.decode(chunk_tokens))
    
    return chunks

def create_document_embeddings(
    document_id: str, 
    text: str, 
    metadata: Optional[Dict] = None
) -> Dict:
    """Create embeddings for a document and store in FAISS index."""
    # Create directory if it doesn't exist
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Always use the document_id for file naming, but store original filename in metadata
    # Split text into chunks
    chunks = chunk_text(text)
    
    # Prepare data for storing
    document_data = {
        "document_id": document_id,
        "chunks": [],
        "metadata": metadata or {}
    }
    
    # Get embeddings for each chunk
    embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            embeddings.append(embedding)
            
            # Store chunk info
            document_data["chunks"].append({
                "chunk_id": f"{document_id}_{i}",
                "text": chunk,
                "embedding_index": i
            })
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")
    
    if not embeddings:
        return {"success": False, "error": "No valid embeddings created"}
    
    # Convert to numpy array for FAISS
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Create or load existing index
    index_path = EMBEDDINGS_DIR / f"{document_id}.index"
    metadata_path = EMBEDDINGS_DIR / f"{document_id}.json"
    
    # Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # Save index and metadata
    faiss.write_index(index, str(index_path))
    
    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(document_data, f)
    
    return {
        "success": True,
        "document_id": document_id,
        "chunks": len(chunks),
        "dimensions": dimension
    }

def search_embeddings(
    document_id: str, 
    query: str, 
    top_k: int = 3
) -> List[Dict]:
    """Search document embeddings for similar chunks."""
    index_path = EMBEDDINGS_DIR / f"{document_id}.index"
    metadata_path = EMBEDDINGS_DIR / f"{document_id}.json"
    
    if not index_path.exists() or not metadata_path.exists():
        return []
    
    # Load index and metadata
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path, "r") as f:
        document_data = json.load(f)
    
    # Get query embedding
    query_embedding = get_embedding(query)
    query_embedding_array = np.array([query_embedding], dtype=np.float32)
    
    # Search for similar chunks
    distances, indices = index.search(query_embedding_array, top_k)
    
    # Prepare results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(document_data["chunks"]):
            chunk = document_data["chunks"][idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(distances[0][i])
            })
    
    return results 

def search_all_documents(query: str, top_k: int = 3) -> List[Dict]:
    """Search across all document embeddings for similar chunks."""
    if not EMBEDDINGS_DIR.exists():
        return []
    
    # Get all metadata files
    metadata_files = list(EMBEDDINGS_DIR.glob("*.json"))
    if not metadata_files:
        return []
    
    # Get query embedding
    query_embedding = get_embedding(query)
    query_embedding_array = np.array([query_embedding], dtype=np.float32)
    
    # Search across all documents
    all_results = []
    for metadata_file in metadata_files:
        document_id = metadata_file.stem  # Get document_id from filename
        
        # Load index and metadata
        index_path = metadata_file.with_suffix(".index")
        if not index_path.exists():
            continue
            
        index = faiss.read_index(str(index_path))
        
        with open(metadata_file, "r") as f:
            document_data = json.load(f)
        
        # Search for similar chunks
        distances, indices = index.search(query_embedding_array, top_k)
        
        # Add results with document context
        for i, idx in enumerate(indices[0]):
            if idx < len(document_data["chunks"]):
                chunk = document_data["chunks"][idx]
                all_results.append({
                    "document_id": document_id,
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": float(distances[0][i]),
                    "metadata": document_data.get("metadata", {})
                })
    
    # Sort by score and take top_k results
    all_results.sort(key=lambda x: x["score"])
    return all_results[:top_k] 

def get_all_documents() -> List[Dict]:
    """Get list of all documents in the documents directory."""
    documents_dir = Path(os.getenv("DOCUMENTS_DIR", "./data/documents"))
    if not documents_dir.exists():
        return []
    
    documents = []
    for file_path in documents_dir.glob("*"):
        if file_path.is_file():
            documents.append({
                "document_id": file_path.stem,
                "filename": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size
            })
    return documents

def get_all_embedded_documents() -> List[str]:
    """Get list of document IDs that have embeddings."""
    if not EMBEDDINGS_DIR.exists():
        return []
    
    embedded_docs = []
    for metadata_file in EMBEDDINGS_DIR.glob("*.json"):
        if metadata_file.with_suffix(".index").exists():
            embedded_docs.append(metadata_file.stem)
    return embedded_docs

def verify_document_embeddings() -> Dict[str, Any]:
    """Verify that all documents have corresponding embeddings."""
    all_docs = get_all_documents()
    embedded_docs = get_all_embedded_documents()
    
    # Find documents without embeddings
    missing_embeddings = [
        doc for doc in all_docs 
        if doc["document_id"] not in embedded_docs
    ]
    
    return {
        "total_documents": len(all_docs),
        "embedded_documents": len(embedded_docs),
        "missing_embeddings": len(missing_embeddings),
        "missing_documents": missing_embeddings,
        "is_complete": len(missing_embeddings) == 0
    } 

def process_missing_embeddings() -> Dict[str, Any]:
    """Process embeddings for any documents that are missing them."""
    verification = verify_document_embeddings()
    if verification["is_complete"]:
        return {
            "message": "All documents are already embedded",
            "verification": verification
        }
    
    results = []
    for doc in verification["missing_documents"]:
        try:
            # Get document content
            content = get_document_content(doc["document_id"])
            if not content:
                results.append({
                    "document_id": doc["document_id"],
                    "success": False,
                    "error": "Could not read document content"
                })
                continue
            
            # Create embeddings
            result = create_document_embeddings(
                doc["document_id"],
                content,
                {"filename": doc["filename"]}
            )
            
            results.append({
                "document_id": doc["document_id"],
                "success": result.get("success", False),
                "chunks": result.get("chunks", 0),
                "error": result.get("error")
            })
            
        except Exception as e:
            results.append({
                "document_id": doc["document_id"],
                "success": False,
                "error": str(e)
            })
    
    # Get updated verification status
    updated_verification = verify_document_embeddings()
    
    return {
        "message": "Processed missing embeddings",
        "results": results,
        "verification": updated_verification
    } 