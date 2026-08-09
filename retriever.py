from state import GraphState
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Initialize the embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# 2. Connect to the local database we created with ingest.py
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)

# 3. Create the retriever interface (fetches top 3 most similar chunks)
db_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def retrieve(state: GraphState) -> dict:
    """
    Retrieve documents from the vector store.
    """
    print("---RETRIEVE---")
    question = state["question"]
    
    # Actually fetch the documents from Chroma
    documents = db_retriever.invoke(question)
    
    return {"documents": documents, "question": question}
