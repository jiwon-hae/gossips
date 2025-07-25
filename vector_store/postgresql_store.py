import asyncpg
import logging
import json

from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from asyncpg.pool import Pool

try:
    from ..utils.env import get_env
    from ..utils.sql import get_sql_loader
    from ..ingestion.chunker.chunk import DocumentChunk
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from utils.env import get_env
    from utils.sql import get_sql_loader
    from ingestion.chunker.chunk import DocumentChunk

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sql_loader = get_sql_loader()
env = get_env()

class PostgreSQLPool:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url if db_url else env.get("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set")

        self.pool: Optional[Pool] = None

    async def initialize(self, min_size: int = 5, max_size: int = 20, max_inactive_connection_lifetime: int = 300, command_timeout: int = 60):
        """Initialize the database connection"""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.db_url,
                min_size=min_size,
                max_size=max_size,
                max_inactive_connection_lifetime=max_inactive_connection_lifetime,
                command_timeout=command_timeout
            )
            logger.info("Database connection pool initialized")

    async def close(self):
        """Close the database connection"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as connection:
            yield connection


postgres_pool = PostgreSQLPool()


async def initilize_db():
    """Initialize the database connection pool"""
    await postgres_pool.initialize()


async def close_db():
    """Close the database connection pool"""
    await postgres_pool.close()


async def create_session(
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout_minutes: int = 60
):
    """
    Create a new session for the user
    
    Args:
        user_id (Optional[str]): The user ID for the session
        metadata (Optional[Dict[str, Any]]): Additional metadata for the session
        timeout_minutes (int): The timeout for the session in minutes
    """
    async with postgres_pool.acquire() as connection:
        expires_at = datetime.now(timezone.utc) + \
            timedelta(minutes=timeout_minutes)

        result = await connection.fetchrow(
            sql_loader.load("sessions", "create.sql"),
            user_id,
            json.dumps(metadata or {}),
            expires_at
        )

        return result["id"]


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get sesssion by ID
    
    Args: 
        session_id : Session UUID
    
    Returns:
        Session data or None if not found/expired
    """
    async with postgres_pool.acquire() as connection:
        result = await connection.fetchrow(
            sql_loader.load("sessions", "get.sql"),
            session_id
        )

        if result:
            return {
                "id": result["id"],
                "user_id": result["user_id"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
                "expires_at": result["expires_at"].isoformat() if result["expires_at"] else None
            }

        return None
    
async def update_session(session_id: str, metadata: Dict[str, Any]) -> bool:
    """
    Update session metadata
    
    Args:
        session_id : Session UUID
        metadata: New metadata to merge
    
    Returns:
        True if updated, False if not found
    """
    async with postgres_pool.acquire() as connection:
        result = await connection.execute(
            sql_loader.load("sessions", "update.sql"),
            session_id,
            json.dumps(metadata)
        )
        
        return result.split()[-1] != "0"


async def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Add a message to the session
    
    Args: 
        session_id: Session UUID
        role: Message role (user/assistant/system)
        content: Message content
        metadata: Optional message metadata
    
    Returns:
        Message ID
    """
    async with postgres_pool.acquire() as connection:
        result = await connection.fetchrow(
            sql_loader.load("message", "insert.sql"),
            session_id,
            role, 
            content,
            json.dumps(metadata or {})
        )
        
        return result["id"]
    

async def save_to_postgres(
        title: str,
        celebrity: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any]
    ) -> str:
        """Save document and chunks to PostgreSQL."""
        try:
            async with postgres_pool.acquire() as connection:
                async with connection.transaction():
                    # Insert document
                    document_result = await connection.fetchrow(
                        sql_loader.load("document", "insert.sql"),
                        title,
                        celebrity,
                        source,
                        content,
                        json.dumps(metadata)
                    )
                    
                    document_id = document_result["id"]
                    
                    # Insert chunks
                    for chunk in chunks:
                        # Convert embedding to PostgreSQL vector string format
                        embedding_data = None
                        if hasattr(chunk, 'embedding') and chunk.embedding:
                            # PostgreSQL vector format: '[1.0,2.0,3.0]' (no spaces after commas)
                            embedding_data = '[' + ','.join(map(str, chunk.embedding)) + ']'
                        
                        await connection.execute(
                            sql_loader.load("chunk", "insert.sql"),
                            document_id,
                            chunk.content,
                            embedding_data,
                            chunk.index,
                            json.dumps(chunk.metadata),
                            chunk.token_count
                        )
                    
                    return document_id
        except Exception as e:
            logger.error(f"Failed to save documents to postgre {title}({e})")
            raise e
            
async def clean_databases():
    """Clean existing data from databases"""
    logger.warning("Clearning existing data from datbases....")
    
    async with postgres_pool.acquire() as connection:
        async with connection.transaction():
                await connection.execute(sql_loader.load("messages", "delete.sql"))
                await connection.execute(sql_loader.load("sessions", "delete.sql"))
                await connection.execute(sql_loader.load("chunks", "delete.sql"))
                await connection.execute(sql_loader.load("documents", "delete.sql"))
    
    logger.info("Cleaned PostgreSQL database")