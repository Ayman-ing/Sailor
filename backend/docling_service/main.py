"""
Standalone Docling Microservice
Deploy this on a powerful machine to handle PDF processing
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import List
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from pydantic import BaseModel

app = FastAPI(title="Docling Processing Service")

# Initialize converter at startup
converter = DocumentConverter()
chunker = HybridChunker(chunk_size=512, chunk_overlap=50)

class Chunk(BaseModel):
    text: str
    chunk_index: int
    token_count: int
    metadata: dict

class ProcessResponse(BaseModel):
    chunks: List[Chunk]
    total_chunks: int

@app.post("/process", response_model=ProcessResponse)
async def process_document(file: UploadFile = File(...)):
    """Process a PDF document and return chunks."""
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    temp_file_path = None
    try:
        # Save uploaded file temporarily
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Convert and chunk
        conversion_result = converter.convert(temp_file_path)
        docling_chunks = chunker.chunk(conversion_result.document)
        
        # Format response
        chunks = []
        for i, chunk in enumerate(docling_chunks):
            chunks.append(Chunk(
                text=chunk.text,
                chunk_index=i,
                token_count=len(chunk.text.split()),
                metadata=chunk.meta.export_json_dict() if chunk.meta else {}
            ))
        
        return ProcessResponse(
            chunks=chunks,
            total_chunks=len(chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "docling-processor"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
