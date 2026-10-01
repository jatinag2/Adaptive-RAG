import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # LLM Settings
    llm_model: str = "gemini-1.5-flash" # updated model
    embedding_model: str = "models/embedding-001" # Or whatever was previously used
    
    # Graph Execution Settings
    max_iterations: int = 3
    
    # Retrieval Settings
    retriever_k: int = 3
    
    # Chunking Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # API Keys
    google_api_key: str = "" # Required in env
    
    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

import logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
