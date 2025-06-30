import pytest
from typing import Dict, Any, Optional

from ingestion.chunker.simple import SimpleChunker
from ingestion.chunker.config import ChunkingConfig
from ingestion.chunker.chunk import DocumentChunk


class TestSimpleChunker:
    """Test cases for SimpleChunker class."""
    
    def test_initialization(self):
        """Test SimpleChunker initialization."""
        config = ChunkingConfig(chunk_size=500)
        chunker = SimpleChunker(config)
        
        assert chunker.config == config
        assert chunker.config.chunk_size == 500
    
    def test_empty_content(self):
        """Test chunking empty content."""
        config = ChunkingConfig()
        chunker = SimpleChunker(config)
        
        chunks = chunker.chunk_document("", "Empty", "empty.txt")
        assert chunks == []
        
        chunks = chunker.chunk_document("   ", "Whitespace", "whitespace.txt")
        assert chunks == []
    
    def test_simple_single_chunk(self):
        """Test chunking content that fits in a single chunk."""
        config = ChunkingConfig(chunk_size=1000)
        chunker = SimpleChunker(config)
        
        content = "This is a short document that should fit in one chunk."
        chunks = chunker.chunk_document(content, "Short Doc", "short.txt")
        
        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].index == 0
        assert chunks[0].metadata["title"] == "Short Doc"
        assert chunks[0].metadata["source"] == "short.txt"
        assert chunks[0].metadata["chunk_method"] == "simple"
        assert chunks[0].metadata["total_chunks"] == 1
    
    def test_multiple_paragraphs_single_chunk(self):
        """Test chunking multiple paragraphs that fit in one chunk."""
        config = ChunkingConfig(chunk_size=1000)
        chunker = SimpleChunker(config)
        
        content = """This is the first paragraph.
        
This is the second paragraph.

This is the third paragraph."""
        
        chunks = chunker.chunk_document(content, "Multi Para", "multi.txt")
        
        assert len(chunks) == 1
        assert "first paragraph" in chunks[0].content
        assert "second paragraph" in chunks[0].content
        assert "third paragraph" in chunks[0].content
    
    def test_multiple_chunks_by_size(self):
        """Test chunking content that requires multiple chunks by size."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = SimpleChunker(config)
        
        # Create content longer than chunk_size
        content = "This is a test document. " * 10  # Much longer than 50 chars
        
        chunks = chunker.chunk_document(content, "Long Doc", "long.txt")
        
        assert len(chunks) > 1
        
        # Check that all chunks except possibly the last are properly sized
        for i, chunk in enumerate(chunks[:-1]):
            assert len(chunk.content) <= config.chunk_size
        
        # Check indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert chunk.metadata["total_chunks"] == len(chunks)
    
    def test_paragraph_splitting(self):
        """Test that content is split on paragraph boundaries when possible."""
        config = ChunkingConfig(chunk_size=100)
        chunker = SimpleChunker(config)
        
        content = """First paragraph is short.

Second paragraph is also quite short.

Third paragraph is longer and might need to be in a separate chunk depending on the chunk size configuration."""
        
        chunks = chunker.chunk_document(content, "Para Test", "para.txt")
        
        assert len(chunks) >= 1
        
        # First chunk should contain complete paragraphs when possible
        first_chunk = chunks[0]
        assert "First paragraph" in first_chunk.content
    
    def test_with_custom_metadata(self):
        """Test chunking with custom metadata."""
        config = ChunkingConfig()
        chunker = SimpleChunker(config)
        
        content = "Test content for metadata testing."
        custom_metadata = {
            "author": "Test Author",
            "category": "test",
            "tags": ["test", "chunking"],
            "version": 1.0
        }
        
        chunks = chunker.chunk_document(content, "Metadata Test", "meta.txt", custom_metadata)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.metadata["title"] == "Metadata Test"
        assert chunk.metadata["source"] == "meta.txt"
        assert chunk.metadata["author"] == "Test Author"
        assert chunk.metadata["category"] == "test"
        assert chunk.metadata["tags"] == ["test", "chunking"]
        assert chunk.metadata["version"] == 1.0
        assert chunk.metadata["chunk_method"] == "simple"
    
    def test_chunk_positioning(self):
        """Test that chunk positions are calculated correctly."""
        config = ChunkingConfig(chunk_size=20, chunk_overlap=5)
        chunker = SimpleChunker(config)
        
        content = "This is a test document for position testing with multiple chunks needed."
        chunks = chunker.chunk_document(content, "Position Test", "pos.txt")
        
        assert len(chunks) > 1
        
        # First chunk should start at 0
        assert chunks[0].start_char == 0
        
        # Check that subsequent chunks have reasonable positions
        # (exact positions depend on paragraph splitting logic)
        for i in range(1, len(chunks)):
            assert chunks[i].start_char >= chunks[i-1].start_char
            assert chunks[i].end_char > chunks[i].start_char
    
    def test_overlap_behavior(self):
        """Test chunk overlap behavior."""
        config = ChunkingConfig(chunk_size=30, chunk_overlap=10)
        chunker = SimpleChunker(config)
        
        # Create predictable content
        content = "A" * 25 + "\n\n" + "B" * 25 + "\n\n" + "C" * 25
        
        chunks = chunker.chunk_document(content, "Overlap Test", "overlap.txt")
        
        if len(chunks) > 1:
            # Check that chunks have the expected metadata
            for chunk in chunks:
                assert chunk.metadata["chunk_overlap"] == config.chunk_overlap
    
    def test_create_chunk_method(self):
        """Test the _create_chunk helper method."""
        config = ChunkingConfig()
        chunker = SimpleChunker(config)
        
        content = "Test content for chunk creation"
        metadata = {"title": "Test", "source": "test.txt", "chunk_method": "simple"}
        
        chunk = chunker._create_chunk(content, 0, 0, len(content), metadata)
        
        assert isinstance(chunk, DocumentChunk)
        assert chunk.content == content
        assert chunk.index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == len(content)
        assert chunk.metadata == metadata
    
    def test_long_single_paragraph(self):
        """Test chunking a single very long paragraph."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = SimpleChunker(config)
        
        # Single paragraph longer than chunk size
        content = "This is a very long single paragraph that will definitely need to be split into multiple chunks because it exceeds the configured chunk size limit by a significant margin."
        
        chunks = chunker.chunk_document(content, "Long Para", "longpara.txt")
        
        assert len(chunks) > 1
        
        # All chunks should have content
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0
    
    def test_mixed_content_types(self):
        """Test chunking content with mixed paragraph lengths."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = SimpleChunker(config)
        
        content = """Short.

This is a medium length paragraph that has some content but is not extremely long.

This is a very long paragraph that definitely exceeds the chunk size limit and should be handled appropriately by the chunking algorithm to ensure proper splitting.

End."""
        
        chunks = chunker.chunk_document(content, "Mixed Content", "mixed.txt")
        
        assert len(chunks) >= 1
        
        # Check that all chunks are properly formed
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert len(chunk.content.strip()) > 0
            assert chunk.metadata["chunk_method"] == "simple"
    
    def test_whitespace_handling(self):
        """Test handling of various whitespace scenarios."""
        config = ChunkingConfig(chunk_size=50)
        chunker = SimpleChunker(config)
        
        content = """   Leading whitespace paragraph.

Paragraph with    internal    spaces.

Trailing whitespace paragraph.   """
        
        chunks = chunker.chunk_document(content, "Whitespace Test", "whitespace.txt")
        
        assert len(chunks) >= 1
        
        # Check that chunks don't have excessive leading/trailing whitespace
        for chunk in chunks:
            assert chunk.content == chunk.content.strip()
    
    def test_inheritance_from_base_chunker(self):
        """Test that SimpleChunker properly inherits from BaseChunker."""
        from ingestion.chunker.chunker import BaseChunker
        
        config = ChunkingConfig()
        chunker = SimpleChunker(config)
        
        assert isinstance(chunker, BaseChunker)
        
        # Test that inherited methods are available
        assert hasattr(chunker, '_validate_chunk_size')
        assert hasattr(chunker, '_filter_valid_chunks')
        
        # Test inherited validation method
        assert chunker._validate_chunk_size("This is a valid chunk") is True
        assert chunker._validate_chunk_size("") is False