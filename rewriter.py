from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import GraphState

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)

system_rewriter_prompt = """You are a question re-writer that converts an input user question to a better version that is optimized 
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewriter_prompt),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]
)

def rewrite_question(state: GraphState) -> dict:
    """
    Transform the query to produce a better question.
    """
    print("Rewrite the query")
    question = state["question"]
    

    question_rewriter = rewrite_prompt | llm | StrOutputParser() 
    rewritten_question = question_rewriter.invoke({"question": question})
    current_count = state.get("search_count", 0)
    
    return {
        "question": rewritten_question, 
        "search_count": current_count + 1
    }
