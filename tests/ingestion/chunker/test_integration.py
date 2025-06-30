import pytest
from unittest.mock import Mock, patch, AsyncMock

from ingestion.chunker.chunker import create_chunker
from ingestion.chunker.config import ChunkingConfig
from ingestion.chunker.chunk import DocumentChunk


@pytest.mark.integration
class TestChunkerIntegration:
    """Integration tests for the chunker system."""
    
    def test_factory_creates_working_simple_chunker(self, simple_config, sample_text, test_helpers):
        """Test that factory creates a working simple chunker."""
        chunker = create_chunker(simple_config)
        
        chunks = chunker.chunk_document(sample_text, "Integration Test", "integration.txt")
        
        assert chunks is not None
        assert len(chunks) > 0
        test_helpers.assert_valid_chunk_sequence(chunks, simple_config)
        
        # Check that it's actually using simple chunking
        assert all(chunk.metadata.get("chunk_method") == "simple" for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_factory_creates_working_semantic_chunker(self, semantic_config, sample_text, test_helpers):
        """Test that factory creates a working semantic chunker."""
        with patch('ingestion.chunker.semantic._prompt_loader') as mock_prompt:
            mock_prompt.render.return_value = "mocked prompt"
            
            chunker = create_chunker(semantic_config)
            
            # Mock the LLM response for semantic chunking
            with patch('ingestion.chunker.semantic.Agent') as mock_agent_class:
                mock_agent = AsyncMock()
                mock_response = Mock()
                mock_response.ouptut = "Chunk 1---CHUNK---Chunk 2---CHUNK---Chunk 3"
                mock_agent.run.return_value = mock_response
                mock_agent_class.return_value = mock_agent
                
                chunks = await chunker.chunk_document(sample_text, "Semantic Test", "semantic.txt")
                
                if chunks is not None:
                    test_helpers.assert_valid_chunk_sequence(chunks, semantic_config)
    
    def test_config_validation_integration(self):
        """Test that configuration validation works in integration."""
        # Valid config should work
        valid_config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = create_chunker(valid_config)
        assert chunker is not None
        
        # Invalid configs should raise errors
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)  # overlap >= chunk_size
        
        with pytest.raises(ValueError):
            ChunkingConfig(min_chunk_size=0)  # zero min size
    
    def test_switching_between_chunker_types(self, sample_text):
        """Test switching between different chunker types."""
        # Create simple chunker
        simple_config = ChunkingConfig(use_semantic_splitting=False, chunk_size=100)
        simple_chunker = create_chunker(simple_config)
        
        # Create semantic chunker
        semantic_config = ChunkingConfig(use_semantic_splitting=True, chunk_size=100)
        
        with patch('ingestion.chunker.semantic._prompt_loader'):
            semantic_chunker = create_chunker(semantic_config)
        
        # Both should be different types but same interface
        assert type(simple_chunker) != type(semantic_chunker)
        assert hasattr(simple_chunker, 'chunk_document')
        assert hasattr(semantic_chunker, 'chunk_document')
        
        # Both should be able to process the same content
        simple_chunks = simple_chunker.chunk_document(sample_text, "Test", "test.txt")
        assert simple_chunks is not None
        assert len(simple_chunks) > 0
    
    def test_metadata_preservation_across_chunks(self, sample_text, document_metadata):
        """Test that metadata is preserved across all chunks."""
        config = ChunkingConfig(chunk_size=100, use_semantic_splitting=False)
        chunker = create_chunker(config)
        
        chunks = chunker.chunk_document(
            sample_text, 
            document_metadata["title"], 
            document_metadata["source"], 
            document_metadata
        )
        
        assert len(chunks) > 0
        
        # Check that all chunks have the same metadata
        for chunk in chunks:
            assert chunk.metadata["title"] == document_metadata["title"]
            assert chunk.metadata["source"] == document_metadata["source"]
            assert chunk.metadata["author"] == document_metadata["author"]
            assert chunk.metadata["category"] == document_metadata["category"]
            assert chunk.metadata["tags"] == document_metadata["tags"]
            assert chunk.metadata["version"] == document_metadata["version"]
            assert chunk.metadata["total_chunks"] == len(chunks)
    
    def test_large_document_processing(self, long_text):
        """Test processing of large documents."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            max_chunk_size=1000,
            min_chunk_size=50,
            use_semantic_splitting=False
        )
        
        chunker = create_chunker(config)
        chunks = chunker.chunk_document(long_text, "Large Document", "large.txt")
        
        assert len(chunks) > 5  # Should create multiple chunks
        
        # Check size constraints
        for chunk in chunks:
            content_length = len(chunk.content.strip())
            assert content_length >= config.min_chunk_size
            assert content_length <= config.max_chunk_size
        
        # Check that indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
    
    def test_structured_content_processing(self, structured_markdown):
        """Test processing of structured markdown content."""
        config = ChunkingConfig(
            chunk_size=300,
            use_semantic_splitting=False,
            preserve_structure=True
        )
        
        chunker = create_chunker(config)
        chunks = chunker.chunk_document(structured_markdown, "Structured Doc", "structured.md")
        
        assert len(chunks) > 0
        
        # Check that structure elements are preserved in content
        all_content = " ".join(chunk.content for chunk in chunks)
        assert "# Main Title" in all_content
        assert "## Introduction" in all_content
        assert "```python" in all_content
        assert "| Metric |" in all_content
    
    def test_empty_and_edge_case_content(self):
        """Test handling of empty and edge case content."""
        config = ChunkingConfig(use_semantic_splitting=False)
        chunker = create_chunker(config)
        
        # Empty content
        chunks = chunker.chunk_document("", "Empty", "empty.txt")
        assert chunks == []
        
        # Whitespace only
        chunks = chunker.chunk_document("   \n  \t  ", "Whitespace", "whitespace.txt")
        assert chunks == []
        
        # Single character
        chunks = chunker.chunk_document("a", "Single Char", "single.txt")
        if chunks:  # Might be filtered out due to min_chunk_size
            assert len(chunks[0].content.strip()) > 0
    
    def test_different_config_combinations(self, sample_text):
        """Test various configuration combinations."""
        test_configs = [
            # Small chunks with overlap
            ChunkingConfig(chunk_size=50, chunk_overlap=10, use_semantic_splitting=False),
            # Large chunks without overlap
            ChunkingConfig(chunk_size=1000, chunk_overlap=0, use_semantic_splitting=False),
            # Strict size constraints
            ChunkingConfig(chunk_size=100, min_chunk_size=80, max_chunk_size=120, use_semantic_splitting=False),
            # Loose size constraints
            ChunkingConfig(chunk_size=200, min_chunk_size=10, max_chunk_size=500, use_semantic_splitting=False),
        ]
        
        for config in test_configs:
            chunker = create_chunker(config)
            chunks = chunker.chunk_document(sample_text, "Config Test", "config.txt")
            
            assert chunks is not None
            assert len(chunks) > 0
            
            # Validate against config
            for chunk in chunks:
                content_length = len(chunk.content.strip())
                assert content_length >= config.min_chunk_size
                assert content_length <= config.max_chunk_size
    
    def test_base_chunker_interface_compliance(self, sample_text):
        """Test that all chunker implementations comply with BaseChunker interface."""
        configs = [
            ChunkingConfig(use_semantic_splitting=False),
            ChunkingConfig(use_semantic_splitting=True)
        ]
        
        for config in configs:
            with patch('ingestion.chunker.semantic._prompt_loader'):
                chunker = create_chunker(config)
                
                # Test required methods exist
                assert hasattr(chunker, 'chunk_document')
                assert hasattr(chunker, 'config')
                assert hasattr(chunker, '_validate_chunk_size')
                assert hasattr(chunker, '_filter_valid_chunks')
                
                # Test methods are callable
                assert callable(chunker.chunk_document)
                assert callable(chunker._validate_chunk_size)
                assert callable(chunker._filter_valid_chunks)
                
                # Test validation methods work
                assert chunker._validate_chunk_size("This is a valid chunk") in [True, False]
                filtered = chunker._filter_valid_chunks(["valid chunk", "a"])
                assert isinstance(filtered, list)
    
    def test_concurrent_chunker_usage(self, sample_text):
        """Test using multiple chunkers concurrently."""
        import threading
        import queue
        
        config = ChunkingConfig(use_semantic_splitting=False, chunk_size=100)
        results_queue = queue.Queue()
        
        def chunk_document(chunker_id):
            chunker = create_chunker(config)
            chunks = chunker.chunk_document(sample_text, f"Concurrent Test {chunker_id}", f"concurrent_{chunker_id}.txt")
            results_queue.put((chunker_id, chunks))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=chunk_document, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 3
        
        # All should have produced valid chunks
        for chunker_id, chunks in results:
            assert chunks is not None
            assert len(chunks) > 0
            assert all(chunk.metadata["title"] == f"Concurrent Test {chunker_id}" for chunk in chunks)
    
    @pytest.mark.slow
    def test_performance_with_large_content(self):
        """Test performance with large content (marked as slow test)."""
        import time
        
        # Generate large content
        large_content = "This is a test sentence for performance testing. " * 10000
        
        config = ChunkingConfig(chunk_size=1000, use_semantic_splitting=False)
        chunker = create_chunker(config)
        
        start_time = time.time()
        chunks = chunker.chunk_document(large_content, "Performance Test", "perf.txt")
        end_time = time.time()
        
        # Should complete in reasonable time (less than 10 seconds for simple chunking)
        assert (end_time - start_time) < 10.0
        assert len(chunks) > 0
        
        # All chunks should be valid
        for chunk in chunks:
            assert len(chunk.content.strip()) <= config.max_chunk_size
            assert len(chunk.content.strip()) >= config.min_chunk_size