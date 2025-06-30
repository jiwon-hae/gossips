import pytest
from unittest.mock import patch, Mock

from ingestion.chunker.chunker import create_chunker, BaseChunker
from ingestion.chunker.config import ChunkingConfig
from ingestion.chunker.simple import SimpleChunker
from ingestion.chunker.semantic import SemanticChunker


class TestCreateChunkerFactory:
    """Test cases for the create_chunker factory function."""
    
    def test_create_semantic_chunker(self):
        """Test creating a semantic chunker when use_semantic_splitting=True."""
        config = ChunkingConfig(use_semantic_splitting=True)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = create_chunker(config)
            
            assert isinstance(chunker, SemanticChunker)
            assert isinstance(chunker, BaseChunker)
            assert chunker.config == config
    
    def test_create_simple_chunker(self):
        """Test creating a simple chunker when use_semantic_splitting=False."""
        config = ChunkingConfig(use_semantic_splitting=False)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = create_chunker(config)
            
            assert isinstance(chunker, SimpleChunker)
            assert isinstance(chunker, BaseChunker)
            assert chunker.config == config
    
    def test_create_simple_chunker_default(self):
        """Test creating chunker with default semantic splitting setting."""
        # Default ChunkingConfig has use_semantic_splitting=True
        config = ChunkingConfig()
        assert config.use_semantic_splitting is True
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = create_chunker(config)
            
            # Should create semantic chunker by default
            assert isinstance(chunker, SemanticChunker)
    
    def test_return_type_annotation(self):
        """Test that factory function returns BaseChunker type."""
        config = ChunkingConfig(use_semantic_splitting=False)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = create_chunker(config)
            
            # Should return BaseChunker interface
            assert isinstance(chunker, BaseChunker)
            assert hasattr(chunker, 'chunk_document')
            assert hasattr(chunker, '_validate_chunk_size')
            assert hasattr(chunker, '_filter_valid_chunks')
    
    def test_config_preservation(self):
        """Test that configuration is properly passed to created chunker."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            max_chunk_size=1000,
            min_chunk_size=50,
            use_semantic_splitting=False,
            preserve_structure=False
        )
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = create_chunker(config)
            
            assert chunker.config.chunk_size == 500
            assert chunker.config.chunk_overlap == 100
            assert chunker.config.max_chunk_size == 1000
            assert chunker.config.min_chunk_size == 50
            assert chunker.config.use_semantic_splitting is False
            assert chunker.config.preserve_structure is False
    
    def test_multiple_chunker_creation(self):
        """Test creating multiple chunkers with different configurations."""
        semantic_config = ChunkingConfig(use_semantic_splitting=True, chunk_size=1000)
        simple_config = ChunkingConfig(use_semantic_splitting=False, chunk_size=500)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            semantic_chunker = create_chunker(semantic_config)
            simple_chunker = create_chunker(simple_config)
            
            assert isinstance(semantic_chunker, SemanticChunker)
            assert isinstance(simple_chunker, SimpleChunker)
            
            assert semantic_chunker.config.chunk_size == 1000
            assert simple_chunker.config.chunk_size == 500
            
            # They should be different instances
            assert semantic_chunker is not simple_chunker
    
    def test_factory_with_edge_case_configs(self):
        """Test factory function with edge case configurations."""
        # Test with minimal configuration
        minimal_config = ChunkingConfig(
            chunk_size=10,
            chunk_overlap=1,
            min_chunk_size=1,
            max_chunk_size=20,
            use_semantic_splitting=False
        )
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = create_chunker(minimal_config)
            
            assert isinstance(chunker, SimpleChunker)
            assert chunker.config.chunk_size == 10
            assert chunker.config.chunk_overlap == 1
    
    def test_factory_with_large_config(self):
        """Test factory function with large configuration values."""
        large_config = ChunkingConfig(
            chunk_size=10000,
            chunk_overlap=1000,
            min_chunk_size=100,
            max_chunk_size=50000,
            use_semantic_splitting=True
        )
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = create_chunker(large_config)
            
            assert isinstance(chunker, SemanticChunker)
            assert chunker.config.chunk_size == 10000
            assert chunker.config.chunk_overlap == 1000
    
    def test_import_behavior(self):
        """Test that imports are handled correctly within the factory function."""
        # The factory function imports SimpleChunker and SemanticChunker locally
        # to avoid circular imports. This test ensures that works correctly.
        
        config = ChunkingConfig(use_semantic_splitting=True)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            # This should not raise import errors
            chunker = create_chunker(config)
            assert chunker is not None
            
        config = ChunkingConfig(use_semantic_splitting=False)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            # This should also not raise import errors
            chunker = create_chunker(config)
            assert chunker is not None
    
    def test_factory_interface_consistency(self):
        """Test that both chunker types provide consistent interface."""
        semantic_config = ChunkingConfig(use_semantic_splitting=True)
        simple_config = ChunkingConfig(use_semantic_splitting=False)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            semantic_chunker = create_chunker(semantic_config)
            simple_chunker = create_chunker(simple_config)
            
            # Both should have the same interface methods
            for chunker in [semantic_chunker, simple_chunker]:
                assert hasattr(chunker, 'chunk_document')
                assert hasattr(chunker, 'config')
                assert hasattr(chunker, '_validate_chunk_size')
                assert hasattr(chunker, '_filter_valid_chunks')
                
                # Test that they can be called (interface compatibility)
                assert callable(chunker.chunk_document)
                assert callable(chunker._validate_chunk_size)
                assert callable(chunker._filter_valid_chunks)
    
    def test_factory_docstring_and_annotations(self):
        """Test that factory function has proper documentation and type annotations."""
        import inspect
        
        # Check function signature
        sig = inspect.signature(create_chunker)
        
        # Should have one parameter: config
        assert len(sig.parameters) == 1
        assert 'config' in sig.parameters
        
        # Check parameter annotation
        config_param = sig.parameters['config']
        assert config_param.annotation == ChunkingConfig
        
        # Check return annotation
        assert sig.return_annotation == BaseChunker
        
        # Check that function has a docstring
        assert create_chunker.__doc__ is not None
        assert "Factory function" in create_chunker.__doc__
    
    def test_same_config_creates_independent_instances(self):
        """Test that using the same config creates independent chunker instances."""
        config = ChunkingConfig(use_semantic_splitting=False, chunk_size=100)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker1 = create_chunker(config)
            chunker2 = create_chunker(config)
            
            # Should be different instances
            assert chunker1 is not chunker2
            
            # But should have the same type and configuration
            assert type(chunker1) == type(chunker2)
            assert chunker1.config == chunker2.config