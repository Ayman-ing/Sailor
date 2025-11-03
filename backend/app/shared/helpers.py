"""Shared helper functions."""

import hashlib
import uuid
from datetime import datetime
from pathlib import Path



def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def generate_hash(content: str) -> str:
    """Generate SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def generate_file_hash(file_content: bytes) -> str:
    """Generate SHA-256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()


def current_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.utcnow()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace unsafe characters
    unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    sanitized = filename
    for char in unsafe_chars:
        sanitized = sanitized.replace(char, '_')
    return sanitized


def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    return Path(filename).suffix.lower().lstrip('.')


def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
    """Check if file extension is allowed."""
    extension = get_file_extension(filename)
    return extension in [ext.lower().lstrip('.') for ext in allowed_extensions]


def bytes_to_mb(size_bytes: int) -> float:
    """Convert bytes to megabytes."""
    return round(size_bytes / (1024 * 1024), 2)


def validate_file_size(size_bytes: int, max_size_mb: int = 50) -> bool:
    """Check if file size is within allowed limit."""
    return bytes_to_mb(size_bytes) <= max_size_mb


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix



def generate_storage_path(user_id: str, document_id: str, filename: str) -> str:
    """Generate storage path for uploaded files."""
    sanitized_name = sanitize_filename(filename)
    return f"users/{user_id}/documents/{document_id}/{sanitized_name}"




