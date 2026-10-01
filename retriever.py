import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import CrossEncoder
from state import GraphState
from config import settings

logger = logging.getLogger(__name__)

# Initialize the embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)

# Connect to the local database we created with ingest.py
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)

# Fetch all docs to build BM25Retriever
try:
    all_docs_data = vectorstore.get()
    all_docs = []
    if all_docs_data and "documents" in all_docs_data:
        for text, meta in zip(all_docs_data["documents"], all_docs_data["metadatas"]):
            all_docs.append(Document(page_content=text, metadata=meta))
            
    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 20 # Retrieve more for reranking
    else:
        bm25_retriever = None
except Exception as e:
    logger.error(f"Could not build BM25 index: {e}")
    bm25_retriever = None

# Create the chroma retriever interface
db_retriever = vectorstore.as_retriever(search_kwargs={"k": 20}) # Top 20 for reranking

# Ensemble Retriever
if bm25_retriever:
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, db_retriever], weights=[0.5, 0.5]
    )
    retriever_to_use = ensemble_retriever
else:
    retriever_to_use = db_retriever

# Initialize Reranker
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def invoke_retriever(question: str):
    return retriever_to_use.invoke(question)

def retrieve(state: GraphState) -> dict:
    """
    Retrieve documents from the vector store and rerank them.
    """
    logger.info("---RETRIEVE & RERANK---")
    question = state["question"]
    
    try:
        # Hybrid retrieval (top ~20)
        documents = invoke_retriever(question)
        logger.info(f"Retrieved {len(documents)} documents. Reranking...")
        
        # Cross-encoder Reranking
        if documents:
            pairs = [[question, doc.page_content] for doc in documents]
            scores = reranker_model.predict(pairs)
            
            # Sort by scores
            scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            
            # Keep top K
            top_k = settings.retriever_k
            documents = [doc for doc, score in scored_docs[:top_k]]
            logger.info(f"Kept top {len(documents)} after reranking.")
            
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        documents = []
    
    return {"documents": documents, "question": question}
