# Advanced Adaptive RAG

A production-ready Adaptive RAG pipeline leveraging LangGraph, Gemini, and ChromaDB. This system routes queries between a local vector store and web search, grades retrieved documents for relevance, cross-encoder reranks them, checks for hallucinations, and ensures the final answer resolves the user's question.

## Features

- **Adaptive Routing**: Intelligently routes queries to `vectorstore` or `web_search` using `DuckDuckGo`.
- **Hybrid Retrieval & Reranking**: Combines dense (Chroma) and sparse (BM25) retrieval via `EnsembleRetriever`, followed by Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) for superior context precision.
- **Self-Correction & Hallucination Checking**: Iteratively grades generation against context and regenerates or rewrites the query if the answer hallucinates or fails to address the question.
- **Graceful Degradation**: Caps generation loops with a `MAX_ITERATIONS` limit, gracefully giving up if grounded information isn't found.
- **Citation Metadata**: Surfaces exact sources and page numbers alongside the final answer.
- **Production Infrastructure**:
  - `config.py` using `pydantic-settings` for centralized configuration.
  - `ingest.py` with content hashing to prevent duplicate vector uploads.
  - FastAPI integration (`api.py`) and Docker support (`Dockerfile` / `docker-compose.yml`).
  - Automated testing with `pytest` and GitHub Actions CI.
  - Retry logic (`tenacity`) and standard Python logging.
- **Evaluation Harness**: Uses `ragas` to score context precision, recall, answer relevancy, and faithfulness (`eval.py`).

## Quickstart

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your Google API key:
```bash
cp .env.example .env
```
Edit `.env` and fill in `GOOGLE_API_KEY`.

### 3. Run the System

You can run the application in three ways:

**A. Streamlit UI (Interactive)**
```bash
streamlit run app.py
```

**B. FastAPI Service**
```bash
uvicorn api:app --reload
```

**C. Docker Compose**
```bash
docker-compose up --build
```

### 4. Evaluate Pipeline (Ragas)
```bash
python eval.py
```

## Architecture

1. **Route**: Determines whether to search the web or vector store.
2. **Retrieve & Rerank**: Fetches top candidates via Hybrid (Dense+Sparse) retrieval, then reranks with a Cross-Encoder.
3. **Grade**: LLM checks if the retrieved context is relevant.
4. **Generate**: Synthesizes an answer.
5. **Validate**: Checks for hallucinations and relevance to the question.
6. **Give Up**: Falls back gracefully if `max_iterations` is reached without a valid answer.
