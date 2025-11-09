"""Custom exceptions for the Sailor application."""

from typing import Any, Dict, Optional


class SailorException(Exception):
    """Base exception for all Sailor application errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


# ==================== Domain Exceptions ====================

class DomainException(SailorException):
    """Base exception for domain layer errors."""
    pass


class ValidationError(DomainException):
    """Raised when domain validation fails."""
    pass


class EntityNotFoundError(DomainException):
    """Raised when an entity is not found."""
    
    def __init__(self, entity_type: str, entity_id: Any):
        message = f"{entity_type} with id '{entity_id}' not found"
        super().__init__(message, details={"entity_type": entity_type, "entity_id": entity_id})


# ==================== Infrastructure Exceptions ====================

class InfrastructureException(SailorException):
    """Base exception for infrastructure layer errors."""
    pass


class VectorStoreError(InfrastructureException):
    """Raised when vector store operations fail."""
    pass


class ExternalAPIError(InfrastructureException):
    """Raised when external API calls fail."""
    
    def __init__(self, service: str, message: str, status_code: Optional[int] = None):
        super().__init__(
            message=f"{service} API error: {message}",
            details={"service": service, "status_code": status_code}
        )


class EmbeddingError(InfrastructureException):
    """Raised when embedding generation fails."""
    pass


# ==================== Document-Specific Exceptions ====================

class DocumentError(SailorException):
    """Base exception for document-related errors."""
    pass


class DocumentNotFoundError(EntityNotFoundError):
    """Raised when a document is not found."""
    
    def __init__(self, document_id: str):
        super().__init__("Document", document_id)


class DocumentUploadError(DocumentError):
    """Raised when document upload fails."""
    pass


class DocumentProcessingError(DocumentError):
    """Raised when document processing fails."""
    
    def __init__(self, document_id: str, reason: str):
        message = f"Failed to process document '{document_id}': {reason}"
        super().__init__(message, details={"document_id": document_id, "reason": reason})


class UnsupportedFileTypeError(DocumentError):
    """Raised when file type is not supported."""
    
    def __init__(self, file_type: str, supported_types: list):
        message = f"File type '{file_type}' is not supported. Supported types: {', '.join(supported_types)}"
        super().__init__(message, details={"file_type": file_type, "supported_types": supported_types})


class PDFExtractionError(DocumentError):
    """Raised when PDF text extraction fails."""
    pass


class ChunkingError(DocumentError):
    """Raised when document chunking fails."""
    pass


# ==================== Storage Exceptions ====================

class StorageError(InfrastructureException):
    """Base exception for storage-related errors."""
    pass
class RepositoryError(StorageError):
    """Raised when repository operations fail."""
    pass

class FileNotFoundError(StorageError):
    """Raised when a file is not found in storage."""
    
    def __init__(self, file_path: str):
        message = f"File not found: {file_path}"
        super().__init__(message, details={"file_path": file_path})


class FileUploadError(StorageError):
    """Raised when file upload fails."""
    pass
