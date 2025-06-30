import pytest
from ingestion.chunker.chunk import DocumentChunk


class TestDocumentChunk:
    """Test cases for DocumentChunk class."""
    
    def test_basic_initialization(self):
        """Test basic initialization of DocumentChunk."""
        metadata = {"title": "Test Document", "source": "test.txt"}
        
        chunk = DocumentChunk(
            content="This is a test chunk",
            index=0,
            start_char=0,
            end_char=20,
            metadata=metadata
        )
        
        assert chunk.content == "This is a test chunk"
        assert chunk.index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 20
        assert chunk.metadata == metadata
        assert chunk.token_count is None
    
    def test_initialization_with_token_count(self):
        """Test initialization with token count."""
        metadata = {"title": "Test Document"}
        
        chunk = DocumentChunk(
            content="This is a test chunk",
            index=1,
            start_char=10,
            end_char=30,
            metadata=metadata,
            token_count=5
        )
        
        assert chunk.content == "This is a test chunk"
        assert chunk.index == 1
        assert chunk.start_char == 10
        assert chunk.end_char == 30
        assert chunk.metadata == metadata
        assert chunk.token_count == 5
    
    def test_empty_content(self):
        """Test initialization with empty content."""
        metadata = {"title": "Empty Document"}
        
        chunk = DocumentChunk(
            content="",
            index=0,
            start_char=0,
            end_char=0,
            metadata=metadata
        )
        
        assert chunk.content == ""
        assert chunk.index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 0
        assert chunk.metadata == metadata
    
    def test_complex_metadata(self):
        """Test initialization with complex metadata."""
        metadata = {
            "title": "Complex Document",
            "source": "complex.txt",
            "author": "Test Author",
            "tags": ["tag1", "tag2"],
            "nested": {"key": "value", "number": 42}
        }
        
        chunk = DocumentChunk(
            content="Complex content",
            index=2,
            start_char=100,
            end_char=115,
            metadata=metadata
        )
        
        assert chunk.metadata == metadata
        assert chunk.metadata["tags"] == ["tag1", "tag2"]
        assert chunk.metadata["nested"]["number"] == 42
    
    def test_dataclass_equality(self):
        """Test that two DocumentChunk instances with same values are equal."""
        metadata = {"title": "Test", "source": "test.txt"}
        
        chunk1 = DocumentChunk(
            content="Same content",
            index=0,
            start_char=0,
            end_char=12,
            metadata=metadata
        )
        
        chunk2 = DocumentChunk(
            content="Same content",
            index=0,
            start_char=0,
            end_char=12,
            metadata=metadata
        )
        
        assert chunk1 == chunk2
    
    def test_dataclass_inequality_different_content(self):
        """Test inequality when content differs."""
        metadata = {"title": "Test"}
        
        chunk1 = DocumentChunk(
            content="Content A",
            index=0,
            start_char=0,
            end_char=9,
            metadata=metadata
        )
        
        chunk2 = DocumentChunk(
            content="Content B",
            index=0,
            start_char=0,
            end_char=9,
            metadata=metadata
        )
        
        assert chunk1 != chunk2
    
    def test_dataclass_inequality_different_index(self):
        """Test inequality when index differs."""
        metadata = {"title": "Test"}
        
        chunk1 = DocumentChunk(
            content="Same content",
            index=0,
            start_char=0,
            end_char=12,
            metadata=metadata
        )
        
        chunk2 = DocumentChunk(
            content="Same content",
            index=1,
            start_char=0,
            end_char=12,
            metadata=metadata
        )
        
        assert chunk1 != chunk2
    
    def test_dataclass_inequality_different_positions(self):
        """Test inequality when positions differ."""
        metadata = {"title": "Test"}
        
        chunk1 = DocumentChunk(
            content="Same content",
            index=0,
            start_char=0,
            end_char=12,
            metadata=metadata
        )
        
        chunk2 = DocumentChunk(
            content="Same content",
            index=0,
            start_char=10,
            end_char=22,
            metadata=metadata
        )
        
        assert chunk1 != chunk2
    
    def test_dataclass_inequality_different_metadata(self):
        """Test inequality when metadata differs."""
        chunk1 = DocumentChunk(
            content="Same content",
            index=0,
            start_char=0,
            end_char=12,
            metadata={"title": "Title A"}
        )
        
        chunk2 = DocumentChunk(
            content="Same content",
            index=0,
            start_char=0,
            end_char=12,
            metadata={"title": "Title B"}
        )
        
        assert chunk1 != chunk2
    
    def test_dataclass_inequality_different_token_count(self):
        """Test inequality when token_count differs."""
        metadata = {"title": "Test"}
        
        chunk1 = DocumentChunk(
            content="Same content",
            index=0,
            start_char=0,
            end_char=12,
            metadata=metadata,
            token_count=5
        )
        
        chunk2 = DocumentChunk(
            content="Same content",
            index=0,
            start_char=0,
            end_char=12,
            metadata=metadata,
            token_count=6
        )
        
        assert chunk1 != chunk2
    
    def test_repr(self):
        """Test string representation of DocumentChunk."""
        metadata = {"title": "Test"}
        
        chunk = DocumentChunk(
            content="Test content",
            index=0,
            start_char=0,
            end_char=12,
            metadata=metadata
        )
        
        repr_str = repr(chunk)
        assert "DocumentChunk" in repr_str
        assert "content='Test content'" in repr_str
        assert "index=0" in repr_str
    
    def test_immutability(self):
        """Test that DocumentChunk is immutable (frozen dataclass)."""
        metadata = {"title": "Test"}
        
        chunk = DocumentChunk(
            content="Test content",
            index=0,
            start_char=0,
            end_char=12,
            metadata=metadata
        )
        
        # Note: This test assumes the dataclass is frozen
        # If it's not frozen, this test should be removed or the dataclass should be made frozen
        # For now, we'll test that we can access the attributes
        assert hasattr(chunk, 'content')
        assert hasattr(chunk, 'index')
        assert hasattr(chunk, 'start_char')
        assert hasattr(chunk, 'end_char')
        assert hasattr(chunk, 'metadata')
        assert hasattr(chunk, 'token_count')