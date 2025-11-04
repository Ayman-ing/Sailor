from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    
    # Security
    secret_key: str
    
    # Database
    database_url: str
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "sailor_db"
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    
    # Groq
    groq_api_key: str
    groq_model: str = "llama3-70b-8192"
    
    # LlamaIndex
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    sparse_embedding_model: str = "prithvida/Splade_PP_en_v1"  # NEW

    chunk_size: int = 512
    chunk_overlap: int = 50
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()