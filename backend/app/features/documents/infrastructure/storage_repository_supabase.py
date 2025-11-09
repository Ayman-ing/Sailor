

from app.core.logger import get_logger
from app.core.supabase_client import get_supabase
from app.core.config import settings
from app.features.documents.domain.repository_interface import StorageRepository
from app.features.documents.domain.value_objects import FileUpload

logger = get_logger(__name__)


class SupabaseStorageRepository(StorageRepository):
    def __init__(self, supabase_client : get_supabase = get_supabase()):
        self.supabase = supabase_client

    async def upload_file(self, file_path: str, file: FileUpload) -> str:
        """Upload a file to Supabase Storage."""
        try:
            response = self.supabase.storage.from_(settings.supabase_bucket_documents).upload(
                path=file_path, 
                file=file.content,
                file_options={"content-type": file.content_type}
            )
            logger.info(f"File uploaded to {response.path} in bucket {settings.supabase_bucket_documents}")
            return response.path

        except Exception as e:
            logger.error(f"Failed to upload file to Supabase Storage: {e}")
            raise
    async def delete_file(self, file_path: str) -> None:
        """Delete a file from Supabase Storage."""
        try:
            response = self.supabase.storage.from_(settings.supabase_bucket_documents).remove([file_path])
            if response.get("error"):
                raise Exception(response["error"]["message"])
            
            logger.info(f"File deleted from {file_path} in bucket {settings.supabase_bucket_documents}")
        
        except Exception as e:
            logger.error(f"Failed to delete file from Supabase Storage: {e}")
            raise

    async def get_file(self, file_path: str) -> bytes:
        """Download a file from Supabase Storage."""
        try:
            response = self.supabase.storage.from_(settings.supabase_bucket_documents).download(file_path)
            if response.get("error"):
                raise Exception(response["error"]["message"])
            
            logger.info(f"File downloaded from {file_path} in bucket {settings.supabase_bucket_documents}")
            return response["data"]
        
        except Exception as e:
            logger.error(f"Failed to download file from Supabase Storage: {e}")
            raise