from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import GraphState


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

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

def format_docs(docs):
    """Helper function to format document objects into a single string."""
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)

def generate(state: GraphState) -> dict:
    """
    Generate an answer using the retrieved, graded documents.
    """
    print("Generate answers")
    question = state["question"]
    documents = state["documents"]
    rag_chain = generate_prompt | llm | StrOutputParser()
    
  
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    
    return {"generation": generation, "question": question}
