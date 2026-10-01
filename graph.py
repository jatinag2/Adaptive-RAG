import logging
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langchain_community.tools import DuckDuckGoSearchRun
from state import GraphState
from router import route_question
from retriever import retrieve
from grader import grade_documents
from rewriter import rewrite_question
from generator import generate
from validator import hallucination_grader, answer_grader
from config import settings

logger = logging.getLogger(__name__)

web_search_tool = DuckDuckGoSearchRun()

def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    logger.info("Decide to generate")
    filtered_documents = state["documents"]
    search_count = state.get("search_count", 0)
    
    if not filtered_documents:
        if search_count >= settings.max_iterations:
            logger.info("DECISION: Max iterations reached. Give up.")
            return "give_up"
        logger.info("DECISION: Not all documents are relevant, rewrite question")
        return "rewrite_question"
    else:
        logger.info("DECISION: Generate")
        return "generate"

def check_hallucinations(state: GraphState) -> str:
    """
    Determines whether the generation is grounded in the document and answers the question.
    """
    logger.info("Checking hallucinations")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    search_count = state.get("search_count", 0)

    doc_texts = [d.page_content for d in documents] 
    hallucination_score = hallucination_grader.invoke(
        {"documents": doc_texts, "generation": generation}
    )
    
    if hallucination_score.binary_score == "yes":
        logger.info("DECISION: Generation is based on the retrieved documents, checking if it resolves the question:")
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_score.binary_score == "yes":
            logger.info("DECISION: Generation resolves the question")
            return "useful"
        else:
            if search_count >= settings.max_iterations:
                logger.info("DECISION: Max iterations reached. Give up.")
                return "give_up"
            logger.info("DECISION: Generation does not resolve the question")
            return "not_useful"
    else:
        if search_count >= settings.max_iterations:
            logger.info("DECISION: Max iterations reached. Give up.")
            return "give_up"
        logger.info("DECISION: Generation suffers from hallucinations, regenerate")
        return "not_supported"

def give_up_node(state: GraphState) -> dict:
    """Fallback node when max iterations are reached."""
    logger.info("Node: give_up")
    return {
        "generation": "I'm sorry, I couldn't find enough grounded information to answer the question accurately.",
        "sources": []
    }

def handle_hallucination_node(state: GraphState) -> dict:
    """Increments the search count before re-generating due to hallucination."""
    logger.info("Node: handle_hallucination (incrementing counter)")
    return {"search_count": state.get("search_count", 0) + 1}

def web_search_node(state: GraphState) -> dict:
    logger.info("Node: web_search")
    question = state["question"]
    try:
        docs = web_search_tool.invoke({"query": question})
        return {
            "documents": [Document(page_content=docs, metadata={"source": "Web Search"})],
            "question": question
        }
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "documents": [Document(page_content="Web search failed.", metadata={"source": "Web Search"})],
            "question": question
        }

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search_node)
workflow.add_node("give_up", give_up_node)
workflow.add_node("handle_hallucination", handle_hallucination_node)

workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "rewrite_question": "rewrite_question",
        "generate": "generate",
        "give_up": "give_up",
    },
)

workflow.add_edge("rewrite_question", "retrieve")

workflow.add_conditional_edges(
    "generate",
    check_hallucinations,
    {
        "not_supported": "handle_hallucination",        
        "not_useful": "rewrite_question",   
        "useful": END,
        "give_up": "give_up",
    },
)

workflow.add_edge("handle_hallucination", "generate")
workflow.add_edge("give_up", END)

app = workflow.compile()
