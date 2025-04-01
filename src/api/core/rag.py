"""RAG (Retrieval Augmented Generation) using OpenAI and FAISS."""
import os
from typing import Dict, List, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv
from .embeddings import search_embeddings, search_all_documents

# Load environment variables
load_dotenv()

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Default model for completions
COMPLETION_MODEL = "gpt-4o-mini"
# Model for query expansion (can use a smaller/faster model)
EXPANSION_MODEL = "gpt-4o-mini"

def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a context string."""
    if not chunks:
        return ""
    
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        formatted_chunks.append(f"[Chunk {i+1}]\n{chunk['text']}\n")
    
    return "\n".join(formatted_chunks)

def expand_query(query: str, num_expansions: int = 3) -> List[str]:
    """Generate expanded queries to improve retrieval."""
    try:
        messages = [
            {"role": "system", "content": (
                "You are a query expansion assistant. Your task is to generate alternative "
                "versions of the user's query that might retrieve additional relevant information. "
                "Generate semantically different but related queries that explore different aspects "
                "or phrasings of the same information need. Return ONLY a numbered list of queries, "
                "no explanations or other text."
            )},
            {"role": "user", "content": f"Original query: '{query}'\n\nGenerate {num_expansions} alternative queries."}
        ]
        
        response = client.chat.completions.create(
            model=EXPANSION_MODEL,
            messages=messages,
            temperature=0.7
        )
        expanded_text = response.choices[0].message.content.strip()
        
        # Parse the expanded queries from the response
        expanded_queries = []
        for line in expanded_text.split('\n'):
            # Remove numbered list formatting and any extra whitespace
            clean_line = line.strip()
            if clean_line:
                # Remove numbering (e.g., "1.", "2.", etc.)
                if clean_line[0].isdigit() and '.' in clean_line[:3]:
                    clean_line = clean_line.split('.', 1)[1].strip()
                # Remove quotes if present
                if clean_line.startswith('"') and clean_line.endswith('"'):
                    clean_line = clean_line[1:-1]
                if clean_line.startswith("'") and clean_line.endswith("'"):
                    clean_line = clean_line[1:-1]
                expanded_queries.append(clean_line)
        
        return expanded_queries[:num_expansions]  # Ensure we return at most num_expansions queries
    
    except Exception as e:
        print(f"Error in query expansion: {str(e)}")
        return []  # Return empty list if expansion fails

def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks based on chunk_id."""
    unique_chunks = {}
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id')
        if chunk_id and chunk_id not in unique_chunks:
            unique_chunks[chunk_id] = chunk
    
    return list(unique_chunks.values())

def generate_answer(
    query: str,
    top_k: int = 3,
    model: str = COMPLETION_MODEL,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """Generate an answer based on retrieved context using query expansion."""
    # Step 1: Expand the original query
    expanded_queries = expand_query(query)

    all_queries = [query] + expanded_queries
    
    # Step 2: Search for relevant chunks using all queries
    all_chunks = []
    for q in all_queries:
        chunks = search_all_documents(q, top_k)
        all_chunks.extend(chunks)
    
    # Step 3: Deduplicate chunks
    unique_chunks = deduplicate_chunks(all_chunks)
    
    if not unique_chunks:
        return {
            "answer": "No relevant information found in any of the documents.",
            "chunks": [],
            "expanded_queries": expanded_queries,
            "success": False
        }
    
    # Step 4: Format context from unique chunks
    context = format_context(unique_chunks)
    
    # Step 5: Create the prompt with context and original query
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the answer cannot be found in the context, say that you don't know based on the available information. "
            "Don't make up information that isn't supported by the context."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    # Step 6: Generate the answer
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        answer = response.choices[0].message.content.strip()
        
        return {
            "answer": answer,
            "chunks": unique_chunks,
            "expanded_queries": expanded_queries,
            "success": True
        }
    
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "chunks": unique_chunks,
            "expanded_queries": expanded_queries,
            "success": False
        } 