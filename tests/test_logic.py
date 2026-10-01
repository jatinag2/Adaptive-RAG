import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from config import settings
from graph import decide_to_generate, check_hallucinations

def test_decide_to_generate_max_iterations():
    # Test that we give up if max iterations is reached and no valid docs
    state = {
        "documents": [],
        "search_count": settings.max_iterations,
        "question": "test"
    }
    decision = decide_to_generate(state)
    assert decision == "give_up"
    
def test_decide_to_generate_rewrite():
    # Test rewriting if no docs and under limit
    state = {
        "documents": [],
        "search_count": settings.max_iterations - 1,
        "question": "test"
    }
    decision = decide_to_generate(state)
    assert decision == "rewrite_question"

def test_decide_to_generate_proceed():
    # Test generation if docs are found
    state = {
        "documents": [Document(page_content="test doc")],
        "search_count": 1,
        "question": "test"
    }
    decision = decide_to_generate(state)
    assert decision == "generate"

@patch('graph.hallucination_grader.invoke')
@patch('graph.answer_grader.invoke')
def test_check_hallucinations_useful(mock_answer, mock_hallucination):
    mock_hallucination.return_value = MagicMock(binary_score="yes")
    mock_answer.return_value = MagicMock(binary_score="yes")
    
    state = {
        "documents": [Document(page_content="test doc")],
        "generation": "test answer",
        "question": "test question",
        "search_count": 1
    }
    
    decision = check_hallucinations(state)
    assert decision == "useful"

@patch('graph.hallucination_grader.invoke')
def test_check_hallucinations_max_limit_not_supported(mock_hallucination):
    mock_hallucination.return_value = MagicMock(binary_score="no")
    
    state = {
        "documents": [Document(page_content="test doc")],
        "generation": "test answer",
        "question": "test question",
        "search_count": settings.max_iterations
    }
    
    decision = check_hallucinations(state)
    assert decision == "give_up"
