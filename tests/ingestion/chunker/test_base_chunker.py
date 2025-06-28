import pytest
from abc import ABC
from typing import List, Dict, Any, Optional

from ingestion.chunker.chunker import BaseChunker
from ingestion.chunker.config import ChunkingConfig
from ingestion.chunker.chunk import DocumentChunk


class ConcreteChunker(BaseChunker):
    """Concrete implementation of BaseChunker for testing."""
    
    def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Simple implementation that creates one chunk per 100 characters."""
        if not content.strip():
            return []
        
        base_metadata = {
            "title": title,
            "source": source,
            **(metadata or {})
        }
        
        chunks = []
        chunk_size = self.config.chunk_size
        
        for i in range(0, len(content), chunk_size):
            chunk_text = content[i:i + chunk_size]
            chunks.append(chunk_text)
        
        return self._create_chunk(chunks, content, base_metadata, "concrete")
    
    def _create_chunk(
        self,
        chunks: List[str],
        original_content: str,
        base_metadata: Dict[str, Any],
        chunk_method: str = "unknown"
    ) -> List[DocumentChunk]:
        """Implementation of abstract method."""
        chunk_objects = []
        current_pos = 0
        
        for i, chunk_text in enumerate(chunks):
            start_pos = original_content.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk_text)
            
            chunk_metadata = {
                **base_metadata,
                "chunk_method": chunk_method,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk_text),
                "chunk_overlap": self.config.chunk_overlap
            }
            
            chunk_objects.append(DocumentChunk(
                content=chunk_text.strip(),
                index=i,
                start_char=start_pos,
                end_char=end_pos,
                metadata=chunk_metadata
            ))
            
            current_pos = end_pos
        
        return chunk_objects


class TestBaseChunker:
    """Test cases for BaseChunker abstract class."""
    
    def test_abstract_class(self):
        """Test that BaseChunker is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseChunker(ChunkingConfig())
    
    def test_concrete_implementation_initialization(self):
        """Test that concrete implementation can be initialized."""
        config = ChunkingConfig(chunk_size=100)
        chunker = ConcreteChunker(config)
        
        assert chunker.config == config
        assert chunker.config.chunk_size == 100
    
    def test_concrete_implementation_chunk_document(self):
        """Test that concrete implementation can chunk documents."""
        config = ChunkingConfig(chunk_size=10)
        chunker = ConcreteChunker(config)
        
        content = "This is a test document with some content."
        chunks = chunker.chunk_document(content, "Test", "test.txt")
        
        assert len(chunks) > 0
        assert isinstance(chunks[0], DocumentChunk)
        assert chunks[0].metadata["title"] == "Test"
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["chunk_method"] == "concrete"
    
    def test_validate_chunk_size_valid(self):
        """Test _validate_chunk_size with valid chunks."""
        config = ChunkingConfig(min_chunk_size=10, max_chunk_size=100)
        chunker = ConcreteChunker(config)
        
        assert chunker._validate_chunk_size("This is a valid chunk") is True
        assert chunker._validate_chunk_size(" " * 50) is True  # 50 spaces
    
    def test_validate_chunk_size_too_small(self):
        """Test _validate_chunk_size with too small chunks."""
        config = ChunkingConfig(min_chunk_size=20, max_chunk_size=100)
        chunker = ConcreteChunker(config)
        
        assert chunker._validate_chunk_size("short") is False
        assert chunker._validate_chunk_size("") is False
    
    def test_validate_chunk_size_too_large(self):
        """Test _validate_chunk_size with too large chunks."""
        config = ChunkingConfig(min_chunk_size=10, max_chunk_size=20)
        chunker = ConcreteChunker(config)
        
        long_text = "This is a very long chunk that exceeds the maximum size limit"
        assert chunker._validate_chunk_size(long_text) is False
    
    def test_validate_chunk_size_edge_cases(self):
        """Test _validate_chunk_size edge cases."""
        config = ChunkingConfig(min_chunk_size=10, max_chunk_size=20)
        chunker = ConcreteChunker(config)
        
        # Exactly min size
        assert chunker._validate_chunk_size("1234567890") is True
        
        # Exactly max size
        assert chunker._validate_chunk_size("12345678901234567890") is True
        
        # One less than min
        assert chunker._validate_chunk_size("123456789") is False
        
        # One more than max
        assert chunker._validate_chunk_size("123456789012345678901") is False
    
    def test_filter_valid_chunks(self):
        """Test _filter_valid_chunks method."""
        config = ChunkingConfig(min_chunk_size=5, max_chunk_size=15)
        chunker = ConcreteChunker(config)
        
        chunks = [
            "short",      # 5 chars - valid
            "hi",         # 2 chars - too small
            "medium text", # 11 chars - valid
            "this is a very long chunk that is too big",  # > 15 chars - too large
            "exactly 15c",  # exactly 15 chars - valid
        ]
        
        valid_chunks = chunker._filter_valid_chunks(chunks)
        
        assert len(valid_chunks) == 3
        assert "short" in valid_chunks
        assert "medium text" in valid_chunks
        assert "exactly 15c" in valid_chunks
        assert "hi" not in valid_chunks
        assert "this is a very long chunk that is too big" not in valid_chunks
    
    def test_filter_valid_chunks_empty_list(self):
        """Test _filter_valid_chunks with empty list."""
        config = ChunkingConfig()
        chunker = ConcreteChunker(config)
        
        valid_chunks = chunker._filter_valid_chunks([])
        assert valid_chunks == []
    
    def test_filter_valid_chunks_all_invalid(self):
        """Test _filter_valid_chunks when all chunks are invalid."""
        config = ChunkingConfig(min_chunk_size=100, max_chunk_size=200)
        chunker = ConcreteChunker(config)
        
        chunks = ["short", "tiny", "small"]
        valid_chunks = chunker._filter_valid_chunks(chunks)
        
        assert valid_chunks == []
    
    def test_create_chunk_with_metadata(self):
        """Test _create_chunk method with custom metadata."""
        config = ChunkingConfig()
        chunker = ConcreteChunker(config)
        
        chunks = ["First chunk", "Second chunk"]
        original_content = "First chunk and Second chunk"
        base_metadata = {"title": "Test", "author": "TestUser"}
        
        result = chunker._create_chunk(chunks, original_content, base_metadata, "test_method")
        
        assert len(result) == 2
        assert result[0].content == "First chunk"
        assert result[1].content == "Second chunk"
        assert result[0].metadata["title"] == "Test"
        assert result[0].metadata["author"] == "TestUser"
        assert result[0].metadata["chunk_method"] == "test_method"
        assert result[0].metadata["total_chunks"] == 2
    
    def test_create_chunk_positioning(self):
        """Test _create_chunk method calculates positions correctly."""
        config = ChunkingConfig()
        chunker = ConcreteChunker(config)
        
        chunks = ["Hello", "world"]
        original_content = "Hello world"
        base_metadata = {"title": "Test"}
        
        result = chunker._create_chunk(chunks, original_content, base_metadata)
        
        assert result[0].start_char == 0
        assert result[0].end_char == 5
        assert result[1].start_char == 6
        assert result[1].end_char == 11
    
    def test_integration_chunk_document_flow(self):
        """Test complete flow of chunking a document."""
        config = ChunkingConfig(chunk_size=20, min_chunk_size=5, max_chunk_size=25)
        chunker = ConcreteChunker(config)
        
        content = "This is a test document with multiple sentences for testing purposes."
        metadata = {"category": "test", "version": 1}
        
        chunks = chunker.chunk_document(content, "Integration Test", "integration.txt", metadata)
        
        assert len(chunks) > 0
        
        # Check first chunk
        first_chunk = chunks[0]
        assert isinstance(first_chunk, DocumentChunk)
        assert first_chunk.metadata["title"] == "Integration Test"
        assert first_chunk.metadata["source"] == "integration.txt"
        assert first_chunk.metadata["category"] == "test"
        assert first_chunk.metadata["version"] == 1
        assert first_chunk.metadata["chunk_method"] == "concrete"
        assert first_chunk.index == 0
        
        # Check that all chunks have sequential indices
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert chunk.metadata["total_chunks"] == len(chunks)