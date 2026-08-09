from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path: str):
    """
    Loads a PDF and splits it into semantic chunks ready for vectorization.
    """
    print(f"LOADING PDF: {file_path}")
    
    
    loader = PyMuPDFLoader(file_path=file_path)
    pages = loader.load()
    
    print(f"Successfully loaded {len(pages)} pages.")

  
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       
        chunk_overlap=200,    
        separators=["\n\n", "\n", ".", " ", ""] 
    )
    
   
    chunks = text_splitter.split_documents(pages)
    
    print(f"Split the PDF into {len(chunks)} chunks.")
    
    if chunks:
        print("\n Sample chunk (First 200 chars)")
        print(f"{chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
        print("\n")
        
    return chunks

if __name__ == "__main__":
   
    sample_pdf_path = r"C:\adaptive_rag\AI_Interview_Question_Bank.pdf"
    
    try:
        document_chunks = load_and_chunk_pdf(sample_pdf_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file '{sample_pdf_path}'. Please check the path.")
