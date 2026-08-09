from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from state import GraphState
from router import route_question
from retriever import retrieve
from grader import grade_documents
from rewriter import rewrite_question
from generator import generate
from validator import hallucination_grader, answer_grader


def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    print("Decide to generate")
    filtered_documents = state["documents"]
    
    if not filtered_documents:
        
        print("DECISION: Not all documents are relevant , rewrite question")
        return "rewrite_question"
    else:
       
        print("DECISION: Generate")
        return "generate"

def check_hallucinations(state: GraphState) -> str:
    """
    Determines whether the generation is grounded in the document and answers the question.
    """
    print("Checking hallucinations")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

 
    doc_texts = [d.page_content for d in documents] 
    hallucination_score = hallucination_grader.invoke(
        {"documents": doc_texts, "generation": generation}
    )
    
    if hallucination_score.binary_score == "yes":
        print("DECISION: Generation is based on the retrieved documents, checking if it resolves the question:")
        
        
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_score.binary_score == "yes":
            print("DECISION: Generation resolves the question")
            return "useful"
        else:
            print("DECISION: Generation does not resolve the question")
            return "not_useful"
    else:
        print("DECISION: Generation suffers from hallucinations, regenerate")
        return "not_supported"


workflow = StateGraph(GraphState)


workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate", generate)


def web_search_node(state: GraphState):
    print("Searching the web now")
    return {
        "documents": [Document(page_content="Web search results...")],
        "question": state["question"]
    }
workflow.add_node("web_search", web_search_node)

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
    },
)


workflow.add_edge("rewrite_question", "retrieve")


workflow.add_conditional_edges(
    "generate",
    check_hallucinations,
    {
        "not_supported": "generate",        
        "not_useful": "rewrite_question",   
        "useful": END,                      
    },
)


app = workflow.compile()
