import os
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from graph import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample Eval Dataset
eval_questions = [
    "What are the different sections in the document?",
    "How does the system handle hallucinations?",
]
eval_ground_truths = [
    ["The document sections typically include introduction, methodology, and conclusion."],
    ["The system checks hallucinations using a grader node and regenerates if it fails."]
]

def run_pipeline(question: str):
    initial_state = {"question": question, "search_count": 0}
    final_state = None
    for output in app.stream(initial_state):
        for key, value in output.items():
            final_state = value
            
    if final_state and "generation" in final_state:
        # Extract contexts from documents
        contexts = [doc.page_content for doc in final_state.get("documents", [])]
        return final_state["generation"], contexts
    return "", []

def run_eval():
    logger.info("Running pipeline for eval dataset...")
    answers = []
    contexts_list = []
    
    for q in eval_questions:
        ans, ctx = run_pipeline(q)
        answers.append(ans)
        contexts_list.append(ctx)
        
    data = {
        "question": eval_questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": eval_ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    logger.info("Starting RAGAS evaluation...")
    # NOTE: ragas uses OpenAI by default, unless configured otherwise.
    # Since this project uses Gemini, one would need to configure Ragas to use Gemini.
    # For now, we define the structure for the harness.
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from config import settings
        
        gemini_llm = ChatGoogleGenerativeAI(model=settings.llm_model)
        gemini_embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
        
        result = evaluate(
            dataset = dataset, 
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            llm=gemini_llm,
            embeddings=gemini_embeddings
        )
        print("\n=== EVALUATION RESULTS ===")
        print(result)
        return result
    except Exception as e:
        logger.error(f"Eval failed: {e}")

if __name__ == "__main__":
    run_eval()
