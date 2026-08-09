from typing import List, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    Represents the state of this adaptive RAG Graph.
    """
    
    question:str
    generation:str
    documents: List[Document]
    search_count:int
    hallucination_score:str
    relevance_score:str
   
