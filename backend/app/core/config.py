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
    
    # Qdrant Vector Database
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_default_collection: str = "sailor_documents"
    
    # Groq LLM
    groq_api_key: str
    groq_model: str = "llama3-70b-8192"
    
    # Embedding Models
    embedding_model: str = "all-MiniLM-L6-v2"  # Dense embedding model
    embedding_dim: int = 384  # Dense embedding dimension
    sparse_embedding_model: str = "prithvida/Splade_PP_en_v1"  # SPLADE sparse model
    
    # Embedding Microservices (HTTP endpoints)
    dense_embedding_url: str = "http://localhost:8001"
    sparse_embedding_url: str = "http://localhost:8002"
    embedding_max_retries: int = 3
    embedding_retry_delay: float = 1.0
    
    # Performance Settings
    embedding_request_batch_size: int = 32  # Batch size for HTTP requests to embedding services
    max_parallel_pages: int = 6  # Concurrent page processing (6 workers for 8-core CPU)
    pages_per_batch: int = 6  # Pages grouped per batch before parallel processing
    embedding_batch_size: int = 32  # Internal batch size for embedding model inference

    # Document Processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size_mb: int = 50  # Maximum upload file size in MB
    
    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    supabase_bucket_documents: str = "documents"
    
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()