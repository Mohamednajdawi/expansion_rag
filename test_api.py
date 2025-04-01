"""Test script for RAG API."""
import requests
import json

def upload_document(file_path: str) -> dict:
    """Upload a document to the API."""
    url = "http://localhost:8000/documents/upload"
    
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "text/plain")}
        response = requests.post(url, files=files)
    
    return response.json()

def ask_question(query: str) -> dict:
    """Ask a question about all documents."""
    url = "http://localhost:8000/qa"
    data = {
        "query": query
    }
    
    response = requests.post(url, json=data)
    return response.json()

def main():
    # Upload the document
    print("Uploading document...")
    result = upload_document("sample_document.txt")
    print(f"Upload result: {json.dumps(result, indent=2)}")
    
    if result.get("success"):
        # Ask some questions
        questions = [
            "What are the main types of machine learning?",
            "What are the key applications of deep learning?",
            "What are the limitations of AI systems?"
        ]
        
        for question in questions:
            print(f"\nAsking question: {question}")
            answer = ask_question(question)
            print(f"Answer: {json.dumps(answer, indent=2)}")

if __name__ == "__main__":
    main() 