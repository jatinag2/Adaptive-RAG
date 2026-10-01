import pytest
import hashlib
from langchain_core.documents import Document
from ingest import generate_chunk_id

def test_generate_chunk_id():
    content = "Hello, world!"
    doc1 = Document(page_content=content, metadata={"source": "test.pdf", "page": 1})
    doc2 = Document(page_content=content, metadata={"source": "test.pdf", "page": 1})
    
    id1 = generate_chunk_id(doc1)
    id2 = generate_chunk_id(doc2)
    
    assert id1 == id2
    
    # Different page or source should change ID
    doc3 = Document(page_content=content, metadata={"source": "test.pdf", "page": 2})
    assert id1 != generate_chunk_id(doc3)
    
    # Different content should change ID
    doc4 = Document(page_content="Different", metadata={"source": "test.pdf", "page": 1})
    assert id1 != generate_chunk_id(doc4)
