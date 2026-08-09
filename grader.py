from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from state import GraphState

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
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

def grade_documents(state: GraphState) -> dict:
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    retrieval_grader = grade_prompt | structured_llm_grader
    filtered_docs = []
    
    for d in documents:
       
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        
        if grade.strip().lower() == "yes":
            print("Grade: DOCUMENT RELEVANT")
            filtered_docs.append(d)
        else:
            print("Grade: DOCUMENT NOT RELEVANT")

            
    return {"documents": filtered_docs, "question": question}
