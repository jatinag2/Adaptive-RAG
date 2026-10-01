import os
import hashlib
import logging
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import settings

logger = logging.getLogger(__name__)

def generate_chunk_id(doc: Document) -> str:
    """Generate a unique ID for a document chunk based on its content."""
    content_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
    return f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('page', 0)}_{content_hash}"

def load_and_chunk_pdf(file_path: str) -> List[Document]:
    """Loads a PDF and splits it into semantic chunks."""
    logger.info(f"LOADING PDF: {file_path}")
    
    loader = PyMuPDFLoader(file_path=file_path)
    pages = loader.load()
    
    logger.info(f"Successfully loaded {len(pages)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,       
        chunk_overlap=settings.chunk_overlap,    
        separators=["\n\n", "\n", ".", " ", ""] 
    )
    
    chunks = text_splitter.split_documents(pages)
    logger.info(f"Split the PDF into {len(chunks)} chunks.")
    
    return chunks

def embed_and_store(chunks: List[Document], persist_directory: str = "./chroma_db"):
    """Embeds documents and stores them in Chroma, avoiding duplicates."""
    if not chunks:
        logger.warning("No chunks to embed and store.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )
    
    # Generate unique IDs for deduplication
    ids = [generate_chunk_id(chunk) for chunk in chunks]
    
    # Filter out chunks that are already in the DB
    existing_ids = set(vectorstore.get(ids=ids)["ids"])
    new_chunks = []
    new_ids = []
    
    for i, doc_id in enumerate(ids):
        if doc_id not in existing_ids:
            new_chunks.append(chunks[i])
            new_ids.append(doc_id)
            
    if new_chunks:
        logger.info(f"Adding {len(new_chunks)} new chunks to Chroma...")
        vectorstore.add_documents(documents=new_chunks, ids=new_ids)
        logger.info("Successfully added chunks to vector store.")
    else:
        logger.info("All chunks already exist in the vector store. Skipping insertion.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        try:
            chunks = load_and_chunk_pdf(pdf_path)
            embed_and_store(chunks)
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
    else:
        logger.info("Usage: python ingest.py <path_to_pdf>")
