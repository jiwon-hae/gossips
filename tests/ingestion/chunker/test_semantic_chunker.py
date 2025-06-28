import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, Optional

from ingestion.chunker.semantic import SemanticChunker
from ingestion.chunker.config import ChunkingConfig
from ingestion.chunker.chunk import DocumentChunk


class TestSemanticChunker:
    """Test cases for SemanticChunker class."""
    
    def test_initialization(self):
        """Test SemanticChunker initialization."""
        config = ChunkingConfig(chunk_size=500, use_semantic_splitting=True)
        
        with patch('ingestion.chunker.semantic.get_embedding_client') as mock_client, \
             patch('ingestion.chunker.semantic.get_ingestion_model') as mock_model:
            
            chunker = SemanticChunker(config)
            
            assert chunker.config == config
            assert chunker.config.chunk_size == 500
            assert chunker.config.use_semantic_splitting is True
            assert chunker.client is not None
            assert chunker.model is not None
    
    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Test chunking empty content."""
        config = ChunkingConfig()
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = SemanticChunker(config)
            
            chunks = await chunker.chunk_document("", "Empty", "empty.txt")
            assert chunks == []
            
            chunks = await chunker.chunk_document("   ", "Whitespace", "whitespace.txt")
            assert chunks == []
    
    @pytest.mark.asyncio
    async def test_semantic_splitting_disabled(self):
        """Test behavior when semantic splitting is disabled."""
        config = ChunkingConfig(use_semantic_splitting=False, chunk_size=1000)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = SemanticChunker(config)
            content = "This is a short document."
            
            # Should return None when semantic splitting is disabled
            chunks = await chunker.chunk_document(content, "Test", "test.txt")
            assert chunks is None
    
    @pytest.mark.asyncio
    async def test_short_content_no_semantic_splitting(self):
        """Test content shorter than chunk_size doesn't trigger semantic splitting."""
        config = ChunkingConfig(use_semantic_splitting=True, chunk_size=1000)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = SemanticChunker(config)
            content = "This is a short document."  # Much shorter than 1000 chars
            
            chunks = await chunker.chunk_document(content, "Short", "short.txt")
            assert chunks is None
    
    @pytest.mark.asyncio
    async def test_semantic_chunking_success(self):
        """Test successful semantic chunking."""
        config = ChunkingConfig(use_semantic_splitting=True, chunk_size=50)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = SemanticChunker(config)
            
            # Mock the semantic chunking method
            mock_chunks = ["First semantic chunk", "Second semantic chunk"]
            chunker._semantic_chunk = AsyncMock(return_value=mock_chunks)
            chunker._create_chunk = Mock(return_value=[
                DocumentChunk("First semantic chunk", 0, 0, 20, {"test": True}),
                DocumentChunk("Second semantic chunk", 1, 20, 42, {"test": True})
            ])
            
            content = "This is a long document that should be semantically chunked into multiple pieces."
            chunks = await chunker.chunk_document(content, "Semantic Test", "semantic.txt")
            
            assert chunks is not None
            assert len(chunks) == 2
            chunker._semantic_chunk.assert_called_once_with(content)
            chunker._create_chunk.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_semantic_chunking_fallback(self):
        """Test fallback when semantic chunking fails."""
        config = ChunkingConfig(use_semantic_splitting=True, chunk_size=50)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'), \
             patch('ingestion.chunker.semantic.logger') as mock_logger:
            
            chunker = SemanticChunker(config)
            
            # Mock semantic chunking to raise an exception
            chunker._semantic_chunk = AsyncMock(side_effect=Exception("Semantic chunking failed"))
            
            content = "This is a long document that should trigger semantic chunking but will fail."
            chunks = await chunker.chunk_document(content, "Fallback Test", "fallback.txt")
            
            # Should log warning and return None (fallback behavior)
            mock_logger.warning.assert_called_once()
            assert "Semantic chunking failed, falling back to simple chunking" in str(mock_logger.warning.call_args)
    
    def test_split_on_structure(self):
        """Test structural splitting functionality."""
        config = ChunkingConfig()
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = SemanticChunker(config)
            
            content = """# Header 1
This is a paragraph.

This is another paragraph.

- List item 1
- List item 2

1. Numbered item 1
2. Numbered item 2

```python
code block
```

| Table | Header |
|-------|--------|
| Cell  | Cell   |"""
            
            sections = chunker._split_on_structure(content)
            
            # Should split into multiple sections
            assert len(sections) > 1
            assert isinstance(sections, list)
            
            # Check that some expected patterns are found
            content_joined = " ".join(sections)
            assert "Header 1" in content_joined
            assert "paragraph" in content_joined
    
    @pytest.mark.asyncio
    async def test_semantic_chunk_processing(self):
        """Test the main semantic chunking logic."""
        config = ChunkingConfig(chunk_size=100, max_chunk_size=200, min_chunk_size=10)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = SemanticChunker(config)
            
            # Mock _split_on_structure to return predictable sections
            chunker._split_on_structure = Mock(return_value=[
                "Short section",
                "This is a medium length section that fits in one chunk",
                "This is a very long section that definitely exceeds the maximum chunk size and will need to be split further using LLM chunking"
            ])
            
            # Mock _split_long_section for the oversized section
            chunker._split_long_section = AsyncMock(return_value=[
                "First part of long section",
                "Second part of long section"
            ])
            
            content = "Mock content for semantic chunking test"
            result = await chunker._semantic_chunk(content)
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Should have called _split_long_section for the oversized section
            chunker._split_long_section.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_split_long_section_with_llm(self):
        """Test LLM-based splitting of long sections."""
        config = ChunkingConfig(min_chunk_size=20, max_chunk_size=100)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'), \
             patch('ingestion.chunker.semantic._prompt_loader') as mock_prompt_loader:
            
            chunker = SemanticChunker(config)
            
            # Mock prompt loader
            mock_prompt_loader.render.return_value = "Mocked prompt"
            
            # Mock pydantic_ai Agent
            mock_response = Mock()
            mock_response.ouptut = "Chunk 1---CHUNK---Chunk 2---CHUNK---Chunk 3"
            
            with patch('ingestion.chunker.semantic.Agent') as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.run.return_value = mock_response
                mock_agent_class.return_value = mock_agent
                
                section = "This is a very long section that needs to be split by the LLM"
                result = await chunker._split_long_section(section)
                
                assert isinstance(result, list)
                # Should have valid chunks based on size constraints
                valid_chunks = [chunk for chunk in result if 20 <= len(chunk) <= 100]
                assert len(valid_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_split_long_section_fallback(self):
        """Test fallback to simple splitting when LLM fails."""
        config = ChunkingConfig(chunk_size=50, min_chunk_size=10, max_chunk_size=100)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'), \
             patch('ingestion.chunker.semantic._prompt_loader') as mock_prompt_loader, \
             patch('ingestion.chunker.semantic.logger') as mock_logger:
            
            chunker = SemanticChunker(config)
            
            # Mock prompt loader to raise exception
            mock_prompt_loader.render.side_effect = Exception("Prompt loading failed")
            
            section = "This is a section that will trigger LLM splitting but will fail and fallback to simple splitting"
            result = await chunker._split_long_section(section)
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Should have logged an error
            mock_logger.error.assert_called_once()
    
    def test_simple_split_method(self):
        """Test the simple splitting fallback method."""
        config = ChunkingConfig(chunk_size=20, chunk_overlap=5, min_chunk_size=5)
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = SemanticChunker(config)
            
            text = "This is a test text that should be split into multiple chunks using simple algorithm."
            result = chunker._simple_split(text)
            
            assert isinstance(result, list)
            assert len(result) > 1
            
            # Check chunk sizes
            for chunk in result[:-1]:  # All but last chunk
                assert len(chunk) <= config.chunk_size + 10  # Allow some tolerance for sentence boundaries
            
            # Check overlap (approximate, due to sentence boundary detection)
            if len(result) > 1:
                # There should be some reasonable splitting
                assert len(result[0]) > 0
                assert len(result[1]) > 0
    
    def test_simple_chunk_method(self):
        """Test the simple chunking wrapper method."""
        config = ChunkingConfig()
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = SemanticChunker(config)
            
            # Mock _simple_split and _create_chunk methods
            chunker._simple_split = Mock(return_value=["Chunk 1", "Chunk 2"])
            chunker._create_chunk = Mock(return_value=[
                DocumentChunk("Chunk 1", 0, 0, 7, {}),
                DocumentChunk("Chunk 2", 1, 7, 14, {})
            ])
            
            content = "Test content"
            metadata = {"title": "Test"}
            
            result = chunker._simple_chunk(content, metadata)
            
            chunker._simple_split.assert_called_once_with(content)
            chunker._create_chunk.assert_called_once_with(["Chunk 1", "Chunk 2"], content, metadata)
            assert result == chunker._create_chunk.return_value
    
    def test_create_chunk_objects_method(self):
        """Test the _create_chunk method for creating DocumentChunk objects."""
        config = ChunkingConfig()
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = SemanticChunker(config)
            
            chunks = ["First chunk content", "Second chunk content"]
            original_content = "First chunk content and Second chunk content"
            base_metadata = {"title": "Test Document", "source": "test.txt"}
            
            result = chunker._create_chunk(chunks, original_content, base_metadata)
            
            assert len(result) == 2
            assert isinstance(result[0], DocumentChunk)
            assert isinstance(result[1], DocumentChunk)
            
            # Check first chunk
            assert result[0].content == "First chunk content"
            assert result[0].index == 0
            assert result[0].metadata["title"] == "Test Document"
            assert result[0].metadata["source"] == "test.txt"
            assert result[0].metadata["chunk_method"] == "semantic"
            assert result[0].metadata["total_chunks"] == 2
            
            # Check second chunk
            assert result[1].content == "Second chunk content"
            assert result[1].index == 1
            assert result[1].metadata["total_chunks"] == 2
    
    def test_inheritance_from_base_chunker(self):
        """Test that SemanticChunker properly inherits from BaseChunker."""
        from ingestion.chunker.chunker import BaseChunker
        
        config = ChunkingConfig()
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = SemanticChunker(config)
            
            assert isinstance(chunker, BaseChunker)
            
            # Test that inherited methods are available
            assert hasattr(chunker, '_validate_chunk_size')
            assert hasattr(chunker, '_filter_valid_chunks')
    
    @pytest.mark.asyncio
    async def test_integration_with_real_content(self):
        """Test integration with realistic content structure."""
        config = ChunkingConfig(
            chunk_size=200,
            max_chunk_size=300,
            min_chunk_size=50,
            use_semantic_splitting=True
        )
        
        with patch('ingestion.chunker.semantic.get_embedding_client'), \
             patch('ingestion.chunker.semantic.get_ingestion_model'):
            
            chunker = SemanticChunker(config)
            
            # Mock the LLM splitting to avoid external dependencies
            chunker._split_long_section = AsyncMock(return_value=[
                "First semantic chunk from long section",
                "Second semantic chunk from long section"
            ])
            
            content = """# Introduction
This is the introduction section of the document.

## Background
Here we provide some background information that is quite detailed and comprehensive.

## Methods
This section describes the methods used in the research. It contains multiple paragraphs with detailed explanations of the methodology and approach taken.

## Results
The results section presents the findings of the study with various subsections and detailed analysis.

## Conclusion
This is the conclusion section that summarizes the key findings and implications."""
            
            chunks = await chunker.chunk_document(content, "Research Paper", "paper.txt")
            
            if chunks is not None:
                assert len(chunks) > 0
                
                # Check that all chunks are properly formed
                for chunk in chunks:
                    assert isinstance(chunk, DocumentChunk)
                    assert len(chunk.content.strip()) >= config.min_chunk_size
                    assert chunk.metadata["chunk_method"] == "semantic"
                    assert chunk.metadata["title"] == "Research Paper"
                    assert chunk.metadata["source"] == "paper.txt"