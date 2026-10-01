import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from state import GraphState
from config import settings

logger = logging.getLogger(__name__)

class RouterQuery(BaseModel):
  """Route a user query to the most appropriate data source."""
  datasource:str = Field(
      description="Given a user question choose to route it to 'vectorstore' or 'web_search'."
  )
  
llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0)
structured_llm_router = llm.with_structured_output(RouterQuery)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def invoke_router(question: str):
    return structured_llm_router.invoke(question)

def route_question(state:GraphState)->str:
    """Route the question to web search or RAG."""
    logger.info("Routing question:")
    question = state["question"]
    
    source = invoke_router(question)
    
    if source.datasource.strip().lower() == "web_search":
        return "web_search"
    else:
        return "vectorstore"
