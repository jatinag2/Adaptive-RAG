import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
from state import GraphState
from config import settings

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0.2)

system_rewriter_prompt = """You are a question re-writer that converts an input user question to a better version that is optimized 
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewriter_prompt),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]
)

question_rewriter = rewrite_prompt | llm | StrOutputParser()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def invoke_rewriter(question: str) -> str:
    return question_rewriter.invoke({"question": question})

def rewrite_question(state: GraphState) -> dict:
    """
    Transform the query to produce a better question.
    """
    logger.info("Rewrite the query")
    question = state["question"]
    
    try:
        rewritten_question = invoke_rewriter(question)
    except Exception as e:
        logger.error(f"Failed to rewrite question, using original: {e}")
        rewritten_question = question

    current_count = state.get("search_count", 0)
    
    return {
        "question": rewritten_question, 
        "search_count": current_count + 1
    }
