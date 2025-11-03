"""Shared interfaces and protocols."""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class Repository(ABC, Generic[T]):
    """Base repository interface."""
    
    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save an entity (create or update)."""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> None:
        """Delete an entity."""
        pass
    
    @abstractmethod
    async def list_by_user(self, user_id: str, skip: int = 0, limit: int = 100) -> List[T]:
        """List entities for a specific user with pagination."""
        pass



