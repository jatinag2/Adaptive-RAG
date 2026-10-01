import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from state import GraphState
from config import settings

logger = logging.getLogger(__name__)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0)
structured_llm_hallucination = llm.with_structured_output(GradeHallucinations)

system_hallucination_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in and supported by the facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_hallucination_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

_hallucination_grader = hallucination_prompt | structured_llm_hallucination

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

structured_llm_answer = llm.with_structured_output(GradeAnswer)

system_answer_prompt = """You are a grader assessing whether an answer addresses / resolves a question. \n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_answer_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

_answer_grader = answer_prompt | structured_llm_answer


class RetryingHallucinationGrader:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def invoke(self, input_dict):
        try:
            return _hallucination_grader.invoke(input_dict)
        except Exception as e:
            logger.error(f"Hallucination grading failed: {e}")
            raise

class RetryingAnswerGrader:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def invoke(self, input_dict):
        try:
            return _answer_grader.invoke(input_dict)
        except Exception as e:
            logger.error(f"Answer grading failed: {e}")
            raise

hallucination_grader = RetryingHallucinationGrader()
answer_grader = RetryingAnswerGrader()
