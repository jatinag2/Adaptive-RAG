from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from state import GraphState

class RouterQuery(BaseModel):
  """Route a user query to the most appropriate data source."""
  datasource:str = Field(
      description="Given a user question choose to route it to 'vectorstore' or 'web_search'."
  )
  
  
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash"  , temperature = 0)
structured_llm_router = llm.with_structured_output(RouterQuery)

def route_question(state:GraphState)->str:
    """Route the question to web search or RAG.
    """
    
    print("Routing question:")
    question = state["question"]
    source = structured_llm_router.invoke(question)
    
    if source.datasource.strip().lower() == "web_search":
     return "web_search"
    else:
     return "vectorstore"
