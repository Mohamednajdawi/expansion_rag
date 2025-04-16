"""Document processing for RAG system."""
import os
import uuid
from typing import Dict, Optional, BinaryIO
from pathlib import Path
import shutil
import logging
import time
from docling.document_converter import DocumentConverter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory to store uploaded documents
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "./data/documents"))

def process_text_document(
    file_content: str,
    filename: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict:
    """Process a text document directly from content."""
    # Create a unique document ID
    document_id = str(uuid.uuid4())
    
    # Create directory if it doesn't exist
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the content
    document_path = DOCUMENTS_DIR / f"{document_id}.txt"
    with open(document_path, "w", encoding="utf-8") as f:
        f.write(file_content)
    
    # Prepare metadata
    doc_metadata = metadata or {}
    if filename:
        doc_metadata["filename"] = filename
    
    return {
        "document_id": document_id,
        "filename": filename or f"{document_id}.txt",
        "path": str(document_path),
        "size": len(file_content),
        "metadata": doc_metadata
    }

def process_pdf_with_retry(document_path: Path, max_retries: int = 3) -> Optional[str]:
    """Process a PDF file with retries using docling."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Processing PDF with docling: {document_path} (attempt {attempt + 1}/{max_retries})")
            
            # Initialize the docling DocumentConverter
            converter = DocumentConverter()
            
            # Convert the document
            result = converter.convert(str(document_path))
            
            # Export to markdown
            content = result.document.export_to_markdown()
            
            if not content or len(content.strip()) == 0:
                if attempt < max_retries - 1:
                    logger.warning(f"No text extracted in attempt {attempt + 1}, retrying...")
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise ValueError("No text could be extracted after all attempts")
            
            logger.info(f"Successfully extracted text from {document_path} using docling")
            return content
                
        except Exception as e:
            logger.error(f"Error processing PDF {document_path} with docling (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retrying
                continue
            else:
                raise
    
    return None

def save_uploaded_file(
    file: BinaryIO,
    filename: str,
    metadata: Optional[Dict] = None
) -> Dict:
    """Save an uploaded file and return its information."""
    # Create a unique document ID
    document_id = str(uuid.uuid4())
    
    # Create directory if it doesn't exist
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get file extension
    _, ext = os.path.splitext(filename)
    if not ext:
        ext = ".txt"
    
    # Save the file
    document_path = DOCUMENTS_DIR / f"{document_id}{ext}"
    with open(document_path, "wb") as f:
        shutil.copyfileobj(file, f)
    
    # Process different file types
    if ext.lower() in (".txt", ".md", ".csv"):
        with open(document_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    elif ext.lower() == ".pdf":
        try:
            content = process_pdf_with_retry(document_path)
            if content is None:
                raise ValueError("Failed to process PDF after all retries")
        except Exception as e:
            logger.error(f"Error processing PDF {document_path}: {str(e)}")
            content = f"Error processing PDF: {str(e)}"
    else:
        content = f"Unsupported file type: {ext}"
    
    # Prepare metadata
    doc_metadata = metadata or {}
    doc_metadata["filename"] = filename
    doc_metadata["file_type"] = ext
    
    return {
        "document_id": document_id,
        "filename": filename,
        "content": content,
        "path": str(document_path),
        "size": os.path.getsize(document_path),
        "metadata": doc_metadata
    }

def get_document_content(document_id: str) -> Optional[str]:
    """Retrieve the content of a stored document."""
    # Look for the document in various extensions
    for ext in [".txt", ".md", ".csv", ".pdf"]:
        document_path = DOCUMENTS_DIR / f"{document_id}{ext}"
        if document_path.exists():
            if ext.lower() == ".pdf":
                try:
                    return process_pdf_with_retry(document_path)
                except Exception as e:
                    logger.error(f"Error reading PDF {document_path}: {str(e)}")
                    return None
            else:
                with open(document_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
    
    logger.error(f"Document not found: {document_id}")
    return None 