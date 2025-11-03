"""Initialize database tables."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import db_manager
from app.core.logger import get_logger



logger = get_logger(__name__)


async def init_database():
    """Initialize database tables."""
    try:
        logger.info("Creating database tables...")
        await db_manager.create_tables()
        logger.info("✅ Database tables created successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        raise
    finally:
        await db_manager.close()


async def drop_database():
    """Drop all database tables."""
    try:
        logger.info("Dropping database tables...")
        await db_manager.drop_tables()
        logger.info("✅ Database tables dropped successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to drop database tables: {e}")
        raise
    finally:
        await db_manager.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "drop":
        asyncio.run(drop_database())
    else:
        asyncio.run(init_database())
