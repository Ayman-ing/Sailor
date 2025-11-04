"""Value objects for the documents feature."""

from dataclasses import dataclass
from typing import List, Optional

from app.shared.helpers import validate_file_extension, bytes_to_mb
from app.shared.exceptions import ValidationError, UnsupportedFileTypeError
from app.features.documents.domain.entities import DEFAULT_USER_ID


@dataclass(frozen=True)
class FileUpload:
    """Value object representing an uploaded file."""
    
    filename: str
    content: bytes
    content_type: str
    
    ALLOWED_EXTENSIONS = ["pdf"]
    MAX_SIZE_MB = 50
    
    def __post_init__(self):
        """Validate file upload."""
        self._validate_extension()
        self._validate_size()
    
    def _validate_extension(self) -> None:
        """Validate file extension."""
        if not validate_file_extension(self.filename, self.ALLOWED_EXTENSIONS):
            raise UnsupportedFileTypeError(
                self.get_extension(),
                self.ALLOWED_EXTENSIONS
            )
    
    def _validate_size(self) -> None:
        """Validate file size."""
        size_mb = bytes_to_mb(len(self.content))
        if size_mb > self.MAX_SIZE_MB:
            raise ValidationError(
                f"File size {size_mb}MB exceeds maximum allowed size of {self.MAX_SIZE_MB}MB"
            )
    
    def get_extension(self) -> str:
        """Get file extension."""
        from app.shared.helpers import get_file_extension
        return get_file_extension(self.filename)
    
    def get_size_mb(self) -> float:
        """Get file size in megabytes."""
        return bytes_to_mb(len(self.content))


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for document chunking."""
    
    chunk_size: int = 1024
    chunk_overlap: int = 50
    
    def __post_init__(self):
        """Validate chunking configuration."""
        if self.chunk_size <= 0:
            raise ValidationError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValidationError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValidationError("chunk_overlap must be less than chunk_size")


@dataclass(frozen=True)
class DocumentFilter:
    """Filter criteria for querying documents."""
    
    user_id: str = DEFAULT_USER_ID
    status: Optional[List[str]] = None
    filename_contains: Optional[str] = None
    skip: int = 0
    limit: int = 100
    
    def __post_init__(self):
        """Validate filter parameters."""
        if self.skip < 0:
            raise ValidationError("skip must be non-negative")
        if self.limit <= 0 or self.limit > 1000:
            raise ValidationError("limit must be between 1 and 1000")
