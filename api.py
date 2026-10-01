import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import os

from graph import app as langgraph_app
from ingest import load_and_chunk_pdf, embed_and_store

logger = logging.getLogger(__name__)

app = FastAPI(title="Adaptive RAG API")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    generation: str
    sources: List[str]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    initial_state = {
        "question": request.query,
        "search_count": 0
    }
    
    final_state = None
    try:
        for output in langgraph_app.stream(initial_state):
            for key, value in output.items():
                final_state = value
    except Exception as e:
        logger.error(f"Error during query execution: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
        
    if final_state and "generation" in final_state:
        return QueryResponse(
            generation=final_state["generation"],
            sources=final_state.get("sources", [])
        )
        
    raise HTTPException(status_code=500, detail="No generation produced.")

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        chunks = load_and_chunk_pdf(temp_path)
        embed_and_store(chunks)
        return {"status": "success", "chunks_processed": len(chunks)}
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
