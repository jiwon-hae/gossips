import asyncio
import logging
import heapq
import hashlib
import time

from datetime import datetime
from typing import List, Optional, Dict, Tuple
from ingestion.chunker.chunk import DocumentChunk
from openai import RateLimitError, APIError

try:
    from ...agent.providers import get_embedding_client, get_embedding_provider
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from agent.providers import get_embedding_client, get_embedding_model

logger = logging.getLogger(__name__)

embedding_client = get_embedding_client()
EMBEDDING_MODEL = get_embedding_model()


class EmbeddingGenerator:
    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.model_configs = {
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191}
        }

        if model not in self.model_configs:
            logger.warning(f"Unknown model {model}, using default config")
            self.config = {"dimensions": 1536, "max_tokens": 8191}
        else:
            self.config = self.model_configs[model]

    async def generate_embedding(self, text: str) -> List[float]:
        # Truncate text if it's too long
        if len(text) > self.config['max_tokens'] * 4:
            text = text[:self.config['max_tokens'] * 4]

        for attempt in range(self.max_retries):
            try:
                response = await embedding_client.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise

                # Exponential backoff for rate limits
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, retrying in {delay}s")
                await asyncio.sleep(delay)
            except APIError as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise

                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Unexpected error generating embedding: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

    async def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("")
                continue

            # Truncate if too long
            if len(text) > self.config["max_tokens"] * 4:
                text = text[:self.config["max_tokens"] * 4]

            processed_texts.append(text)

        for attempt in range(self.max_retries):
            try:
                response = await embedding_client.embeddings.create(
                    model=self.model,
                    input=processed_texts
                )

                return [data.embedding for data in response.data]

            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise

                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, retrying batch in {delay}s")
                await asyncio.sleep(delay)

            except APIError as e:
                logger.error(f"OpenAI API error in batch: {e}")
                if attempt == self.max_retries - 1:
                    # Fallback to individual processing
                    return await self._process_individually(processed_texts)
                await asyncio.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Unexpected error in batch embedding: {e}")
                if attempt == self.max_retries - 1:
                    return await self._process_individually(processed_texts)
                await asyncio.sleep(self.retry_delay)

    async def _process_individually(self, texts: List[str]):
        """
        Process texts individually as fallback in case batch generation fails
        """
        embeddings = []

        for text in texts:
            try:
                if not text or not text.strip():
                    embeddings.append([0.0] * self.config["dimensions"])
                    continue

                embedding = await self.generate_embedding(text)
                embeddings.append(embedding)

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                embeddings.append([0.0] * self.config["dimensions"])

        return embeddings

    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        process_callback: Optional[callable] = None
    ):
        """
        Generates embedding from document chunks
        """

        if not chunks:
            return chunks

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        embedded_chunks = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i+self.batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]

            try:
                embeddings = await self.generate_embeddings_batch(batch_texts)

                for chunk, embedding in zip(batch_chunks, embeddings):
                    embedded_chunk = DocumentChunk(
                        content=chunk.content,
                        index=chunk.index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        metadata={
                            **chunk.metadata,
                            "embedding_model": self.model,
                            "embedding_generated_at": datetime.now().isoformat()
                        },
                        token_count=chunk.token_count
                    )
                    
                    embedded_chunk.embedding = embedding
                    embedded_chunks.append(embedded_chunk)
                
                current_batch = (i // self.batch_size) + 1
                if process_callback:
                    process_callback(current_batch, total_batches)
                logger.info(f"Processed batch {current_batch}/{total_batches}")

            except Exception as e:
                logger.error(f"Failed to process batch {i//self.batch_size + 1}: {e}")
                
                # Add chunks without embeddings as fallback
                for chunk in batch_chunks:
                    chunk.metadata.update({
                        "embedding_error": str(e),
                        "embedding_generated_at": datetime.now().isoformat()
                    })
                    chunk.embedding = [0.0] * self.config["dimensions"]
                    embedded_chunks.append(chunk)
        
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    
    async def embed_query(self, query: str) -> List[float]:
        """ Genearte embedding for a search query"""
        return await self.generate_embedding(query)
    
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        return self.config["dimensions"]    


class EmbeddingCache:
    """LRU Cache for embeddings using heapq for efficient eviction."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, List[float]] = {}
        self.max_size = max_size
        self.access_counter = 0
        # Min-heap: (access_time, text_hash) - oldest access time first
        self.access_heap: List[Tuple[int, str]] = []
        # Track current access time for each key
        self.key_access_time: Dict[str, int] = {}
        
    def _hash_text(self, text: str) -> str:
        """Create a hash of the text for use as cache key."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if it exists."""
        text_hash = self._hash_text(text)
        
        if text_hash in self.cache:
            # Update access time
            self.access_counter += 1
            self.key_access_time[text_hash] = self.access_counter
            # Add new access record to heap
            heapq.heappush(self.access_heap, (self.access_counter, text_hash))
            return self.cache[text_hash]
        
        return None
    
    def put(self, text: str, embedding: List[float]):
        """Store embedding in cache with LRU eviction if necessary."""
        text_hash = self._hash_text(text)
        
        # If key already exists, just update it
        if text_hash in self.cache:
            self.cache[text_hash] = embedding
            self.access_counter += 1
            self.key_access_time[text_hash] = self.access_counter
            heapq.heappush(self.access_heap, (self.access_counter, text_hash))
            return
        
        # If cache is full, evict least recently used items
        while len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Add new item
        self.cache[text_hash] = embedding
        self.access_counter += 1
        self.key_access_time[text_hash] = self.access_counter
        heapq.heappush(self.access_heap, (self.access_counter, text_hash))
    
    def _evict_lru(self):
        """Evict the least recently used item from cache using heapq."""
        while self.access_heap:
            access_time, text_hash = heapq.heappop(self.access_heap)
            
            # Check if this is the most recent access for this key
            if (text_hash in self.key_access_time and 
                self.key_access_time[text_hash] == access_time):
                # This is the LRU item, remove it
                if text_hash in self.cache:
                    del self.cache[text_hash]
                del self.key_access_time[text_hash]
                break
            # Otherwise, this is a stale entry, just continue to next
    
    def clear(self):
        """Clear all cached embeddings."""
        self.cache.clear()
        self.access_heap.clear()
        self.key_access_time.clear()
        self.access_counter = 0
    
    def size(self) -> int:
        """Return current cache size."""
        return len(self.cache)
    
    def is_full(self) -> bool:
        """Check if cache is at maximum capacity."""
        return len(self.cache) >= self.max_size
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'heap_size': len(self.access_heap),
            'access_counter': self.access_counter
        }



def create_embedder(
    model: str = EMBEDDING_MODEL,
    use_cache: bool = True,
    cache_size: int = 1000,
    **kwargs
) -> EmbeddingGenerator:
    """Create an embedding generator, optionally with caching.
    
    Args:
        model: The embedding model to use
        use_cache: Whether to enable LRU caching
        cache_size: Maximum number of embeddings to cache
        **kwargs: Additional arguments passed to the embedding generator
    
    Returns:
        An embedding generator instance
    """
    embedder = EmbeddingGenerator(model=model, **kwargs)
    
    if use_cache:
        cache = EmbeddingCache()
        original_generate = embedder.generate_embedding
        
        async def cache_generate(text: str) -> List[float]:
            cached = cache.get(text)
            if cached is not None:
                return cached
            
            embedding = await original_generate(text)
            cache.put(text, embedding)
            return embedding

        embedder.generate_embedding = cache_generate
    
    return embedder