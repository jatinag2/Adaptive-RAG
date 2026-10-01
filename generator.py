import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
from state import GraphState
from config import settings

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0)

system_generator_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise."""

generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_generator_prompt),
        ("human", "Context: \n\n {context} \n\n Question: {question}"),
    ]
)

rag_chain = generate_prompt | llm | StrOutputParser()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def invoke_generator(context: str, question: str):
    return rag_chain.invoke({"context": context, "question": question})

def format_docs(docs):
    """Helper function to format document objects into a single string."""
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)

def generate(state: GraphState) -> dict:
    """
    Generate an answer using the retrieved, graded documents.
    """
    logger.info("Generate answers")
    question = state["question"]
    documents = state["documents"]
    
    context = format_docs(documents)
    
    try:
        generation = invoke_generator(context, question)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        generation = "I'm sorry, I encountered an error while trying to generate an answer."

    # Extract sources from documents
    sources = []
    for doc in documents:
        if hasattr(doc, "metadata"):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            if source == "Web Search":
                sources.append(f"Web Search")
            else:
                sources.append(f"{source} (Page {page})")
    
    # Deduplicate sources while preserving order
    unique_sources = []
    for s in sources:
        if s not in unique_sources:
            unique_sources.append(s)

    return {"generation": generation, "question": question, "sources": unique_sources}
