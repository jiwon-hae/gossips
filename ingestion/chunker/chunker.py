from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ingestion.chunker.config import ChunkingConfig
from ingestion.chunker.chunk import DocumentChunk


class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.
    
    This class defines the interface that all chunker implementations must follow,
    providing a consistent API for different chunking strategies (simple, semantic, etc.).
    
    Attributes:
        config (ChunkingConfig): Configuration parameters for chunking
    """
    
    def __init__(self, config: ChunkingConfig):
        """
        Initialize the chunker with configuration.
        
        Args:
            config (ChunkingConfig): Configuration parameters for chunking
        """
        self.config = config
    
    @abstractmethod
    def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces based on the implementation strategy.
        
        This is the main method that subclasses must implement to provide their
        specific chunking logic.
        
        Args:
            content (str): The document content to be chunked
            title (str): The title of the document
            source (str): The source/path of the document
            metadata (Optional[Dict[str, Any]]): Additional metadata to include
        
        Returns:
            List[DocumentChunk]: List of document chunks with metadata
        
        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        pass
    
    @abstractmethod
    def _create_chunk(
        self,
        chunks: List[str],
        original_content: str,
        base_metadata: Dict[str, Any],
        chunk_method: str = "unknown"
    ) -> List[DocumentChunk]:
        """
        Create DocumentChunk objects from text chunks.
        
        This helper method converts raw text chunks into DocumentChunk objects
        with proper positioning and metadata.
        
        Args:
            chunks (List[str]): List of chunk texts
            original_content (str): Original document content for position calculation
            base_metadata (Dict[str, Any]): Base metadata to include in each chunk
            chunk_method (str): The chunking method used (for metadata)
        
        Returns:
            List[DocumentChunk]: List of DocumentChunk objects
        """
        pass
    
    def _validate_chunk_size(self, chunk: str) -> bool:
        """
        Validate if a chunk meets the size requirements.
        
        Args:
            chunk (str): The chunk text to validate
        
        Returns:
            bool: True if chunk size is within acceptable range
        """
        chunk_len = len(chunk.strip())
        return (self.config.min_chunk_size <= chunk_len <= self.config.max_chunk_size)
    
    def _filter_valid_chunks(self, chunks: List[str]) -> List[str]:
        """
        Filter chunks to only include those that meet size requirements.
        
        Args:
            chunks (List[str]): List of chunk texts to filter
        
        Returns:
            List[str]: List of valid chunks
        """
        return [chunk for chunk in chunks if self._validate_chunk_size(chunk)]


def create_chunker(config: ChunkingConfig) -> BaseChunker:
    """
    Factory function to create appropriate chunker based on configuration.
    
    Args:
        config (ChunkingConfig): Configuration parameters
    
    Returns:
        BaseChunker: Appropriate chunker implementation
    """
    # Import here to avoid circular imports
    from ingestion.chunker.semantic import SemanticChunker
    from ingestion.chunker.simple import SimpleChunker
    
    if config.use_semantic_splitting:
        return SemanticChunker(config)
    return SimpleChunker(config)