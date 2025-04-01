# RAG API

A simple Retrieval Augmented Generation (RAG) API using OpenAI for embeddings and generation, with FAISS for vector similarity search.

## Features

- Document upload and processing
- Text embedding using OpenAI's embedding models
- Vector similarity search with FAISS
- Question answering using RAG with OpenAI models
- FastAPI for a robust and well-documented API

## Requirements

- Python 3.12+
- OpenAI API key

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install the package in development mode:
   ```bash
   pip install -e .
   ```
4. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   EMBEDDINGS_DIR=./data/embeddings
   DOCUMENTS_DIR=./data/documents
   ```

## Usage

### Running the API

```bash
# Run the API server
api
```

Or alternatively:

```bash
python -m uvicorn api.app:app --reload
```

The API will be available at http://localhost:8000

### API Documentation

Once the API is running, you can access the auto-generated documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Endpoints

- `POST /documents/upload`: Upload a document file
- `POST /documents/text`: Process a text document directly
- `GET /documents/{document_id}`: Get document information
- `POST /qa`: Answer a question using RAG

## Example

1. Upload a document:

   ```bash
   curl -X POST -F "file=@your_document.txt" http://localhost:8000/documents/upload
   ```

2. Ask a question:
   ```bash
   curl -X POST -H "Content-Type: application/json" \
     -d '{"document_id": "your_document_id", "query": "Your question here?"}' \
     http://localhost:8000/qa
   ```

## License

MIT
