"""Supabase client for storage and database operations."""
from supabase import create_client, Client
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class SupabaseClient:
    """Supabase client wrapper with singleton pattern."""
    
    _instance: Client | None = None
    
    @classmethod
    def get_client(cls) -> Client:
        """Get or create Supabase client instance.
        
        Returns:
            Supabase client instance with service role key (bypasses RLS)
        """
        if cls._instance is None:
            logger.info("Initializing Supabase client")
            cls._instance = create_client(
                supabase_url=settings.supabase_url,
                supabase_key=settings.supabase_service_role_key  # Use service role for backend
            )
            logger.info("Supabase client initialized successfully")
        return cls._instance


# Global instance getter
def get_supabase() -> Client:
    """Get Supabase client instance.
    
    Returns:
        Supabase client configured with service role key
    """
    return SupabaseClient.get_client()
