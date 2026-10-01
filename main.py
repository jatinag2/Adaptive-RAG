import os
import logging
from graph import app

# Set up simple logging for CLI execution
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_rag_agent(query: str):
    print(f"\n{'='*50}\nStarting Adaptive RAG for query:\n'{query}'\n{'='*50}\n")
    
    initial_state = {
        "question": query,
        "search_count": 0
    }
    
    final_state = None
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"\nNode Executed: [{key}]")
            
            if "documents" in value:
                print(f"-> Documents in state: {len(value['documents'])}")
            if "search_count" in value:
                print(f"-> Iteration count: {value['search_count']}")
            final_state = value
                
    print(f"\n{'='*50}")
    print("FINAL GENERATION:")
    print("="*50)
    
    if final_state and "generation" in final_state:
        print("\n" + final_state["generation"])
        
        sources = final_state.get("sources", [])
        if sources:
            print("\nSOURCES:")
            for s in sources:
                print(f" - {s}")
    else:
        print("No generation produced. The graph may have hit a fallback limit.")

if __name__ == "__main__":
    test_query_1 = "what are the different sections in the document"
    test_query_2 = "What were the major news headlines today?"
    
    run_rag_agent(test_query_1)
    run_rag_agent(test_query_2)
