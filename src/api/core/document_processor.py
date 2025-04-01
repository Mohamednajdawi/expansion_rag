"""Document processing for RAG system."""
import os
import uuid
from typing import Dict, Optional, BinaryIO
from pathlib import Path
import shutil
import pdfplumber
import logging
import time

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
    """Process a PDF file with retries."""
    for attempt in range(max_retries):
        try:
            with pdfplumber.open(document_path) as pdf:
                logger.info(f"Processing PDF: {document_path} (attempt {attempt + 1}/{max_retries})")
                text_parts = []
                
                # Get total pages
                total_pages = len(pdf.pages)
                logger.info(f"PDF has {total_pages} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Add a small delay between pages to avoid potential issues
                        if page_num > 1:
                            time.sleep(0.1)
                            
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                            logger.info(f"Successfully extracted text from page {page_num}/{total_pages}")
                        else:
                            logger.warning(f"No text extracted from page {page_num}/{total_pages}")
                    except Exception as page_error:
                        logger.error(f"Error extracting text from page {page_num}/{total_pages}: {str(page_error)}")
                        continue
                
                if not text_parts:
                    if attempt < max_retries - 1:
                        logger.warning(f"No text extracted in attempt {attempt + 1}, retrying...")
                        time.sleep(1)  # Wait before retrying
                        continue
                    else:
                        raise ValueError("No text could be extracted from any page after all attempts")
                
                content = "\n".join(text_parts)
                logger.info(f"Successfully extracted {len(text_parts)} pages of text from {document_path}")
                return content
                
        except Exception as e:
            logger.error(f"Error processing PDF {document_path} (attempt {attempt + 1}/{max_retries}): {str(e)}")
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