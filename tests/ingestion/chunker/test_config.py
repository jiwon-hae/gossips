import pytest
from ingestion.chunker.config import ChunkingConfig


class TestChunkingConfig:
    """Test cases for ChunkingConfig class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ChunkingConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_chunk_size == 2000
        assert config.min_chunk_size == 100
        assert config.use_semantic_splitting is True
        assert config.preserve_structure is True
    
    def test_custom_values(self):
        """Test initialization with custom values."""
        config = ChunkingConfig(
            chunk_size=1500,
            chunk_overlap=300,
            max_chunk_size=3000,
            min_chunk_size=200,
            use_semantic_splitting=False,
            preserve_structure=False
        )
        
        assert config.chunk_size == 1500
        assert config.chunk_overlap == 300
        assert config.max_chunk_size == 3000
        assert config.min_chunk_size == 200
        assert config.use_semantic_splitting is False
        assert config.preserve_structure is False
    
    def test_validation_overlap_greater_than_chunk_size(self):
        """Test validation fails when overlap >= chunk_size."""
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)
    
    def test_validation_overlap_equal_to_chunk_size(self):
        """Test validation fails when overlap equals chunk_size."""
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            ChunkingConfig(chunk_size=200, chunk_overlap=200)
    
    def test_validation_zero_min_chunk_size(self):
        """Test validation fails when min_chunk_size is zero."""
        with pytest.raises(ValueError, match="Minimum chunk size must be positive"):
            ChunkingConfig(min_chunk_size=0)
    
    def test_validation_negative_min_chunk_size(self):
        """Test validation fails when min_chunk_size is negative."""
        with pytest.raises(ValueError, match="Minimum chunk size must be positive"):
            ChunkingConfig(min_chunk_size=-10)
    
    def test_valid_configuration(self):
        """Test that valid configurations pass validation."""
        # This should not raise any exceptions
        config = ChunkingConfig(
            chunk_size=1000,
            chunk_overlap=200,
            max_chunk_size=2000,
            min_chunk_size=50
        )
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
    
    def test_edge_case_overlap_one_less_than_chunk_size(self):
        """Test edge case where overlap is one less than chunk_size."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=99)
        assert config.chunk_overlap == 99
    
    def test_edge_case_min_chunk_size_one(self):
        """Test edge case where min_chunk_size is 1."""
        config = ChunkingConfig(min_chunk_size=1)
        assert config.min_chunk_size == 1
    
    def test_dataclass_equality(self):
        """Test that two ChunkingConfig instances with same values are equal."""
        config1 = ChunkingConfig(chunk_size=500, chunk_overlap=100)
        config2 = ChunkingConfig(chunk_size=500, chunk_overlap=100)
        
        assert config1 == config2
    
    def test_dataclass_inequality(self):
        """Test that two ChunkingConfig instances with different values are not equal."""
        config1 = ChunkingConfig(chunk_size=500, chunk_overlap=100)
        config2 = ChunkingConfig(chunk_size=600, chunk_overlap=100)
        
        assert config1 != config2
    
    def test_repr(self):
        """Test string representation of ChunkingConfig."""
        config = ChunkingConfig(chunk_size=500)
        repr_str = repr(config)
        
        assert "ChunkingConfig" in repr_str
        assert "chunk_size=500" in repr_str
        assert "chunk_overlap=200" in repr_str  # default value