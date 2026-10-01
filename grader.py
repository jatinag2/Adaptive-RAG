import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from state import GraphState
from config import settings

logger = logging.getLogger(__name__)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_grader_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grader_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def invoke_grader(question: str, document: str):
    return retrieval_grader.invoke({"question": question, "document": document})

def grade_documents(state: GraphState) -> dict:
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    logger.info("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    
    for d in documents:
        try:
            score = invoke_grader(question, d.page_content)
            grade = score.binary_score
            
            if grade.strip().lower() == "yes":
                logger.info("Grade: DOCUMENT RELEVANT")
                filtered_docs.append(d)
            else:
                logger.info("Grade: DOCUMENT NOT RELEVANT")
        except Exception as e:
            logger.error(f"Error grading document: {e}")
            
    return {"documents": filtered_docs, "question": question}
