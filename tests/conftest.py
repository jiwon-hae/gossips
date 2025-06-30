import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock external dependencies at import time
def mock_missing_modules():
    """Mock missing external modules to prevent import errors."""
    missing_modules = [
        'openai',
        'pydantic_ai',
        'pydantic_ai.models.openai', 
        'pydantic_ai.providers.openai',
        'jinja2',
    ]
    
    for module_name in missing_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = Mock()
    
    # Mock openai module
    if 'openai' not in sys.modules or isinstance(sys.modules['openai'], Mock):
        openai_mock = Mock()
        openai_mock.OpenAI = Mock
        openai_mock.AsyncOpenAI = Mock
        sys.modules['openai'] = openai_mock
    
    # Mock pydantic_ai module
    if 'pydantic_ai' not in sys.modules or isinstance(sys.modules['pydantic_ai'], Mock):
        pydantic_ai_mock = Mock()
        pydantic_ai_mock.Agent = Mock
        sys.modules['pydantic_ai'] = pydantic_ai_mock
    
    # Mock pydantic_ai.models.openai
    if 'pydantic_ai.models.openai' not in sys.modules:
        models_openai_mock = Mock()
        models_openai_mock.OpenAIModel = Mock
        sys.modules['pydantic_ai.models.openai'] = models_openai_mock
    
    # Mock pydantic_ai.providers.openai
    if 'pydantic_ai.providers.openai' not in sys.modules:
        providers_openai_mock = Mock()
        providers_openai_mock.OpenAIProvider = Mock
        sys.modules['pydantic_ai.providers.openai'] = providers_openai_mock
    
    # Mock jinja2
    if 'jinja2' not in sys.modules or isinstance(sys.modules['jinja2'], Mock):
        jinja2_mock = Mock()
        jinja2_mock.Environment = Mock
        jinja2_mock.FileSystemLoader = Mock
        sys.modules['jinja2'] = jinja2_mock

# Apply mocks immediately
mock_missing_modules()


@pytest.fixture
def sample_text():
    """Fixture providing sample text for testing chunkers."""
    return """This is the first paragraph of the sample text. It contains some basic information and serves as an introduction to the content.

This is the second paragraph. It provides additional details and continues the narrative flow of the document.

This is a longer third paragraph that contains significantly more text than the previous paragraphs. It is designed to test how chunkers handle content that might need to be split due to size constraints. The paragraph includes multiple sentences with varying lengths to provide a realistic test case for the chunking algorithms.

# This is a markdown header

Following the header, we have another paragraph that demonstrates how structured content should be handled by the chunking system.

- This is a bullet point
- This is another bullet point  
- And a third bullet point for testing list handling

1. This is a numbered list item
2. This is the second numbered item
3. And the third numbered item

```python
# This is a code block
def example_function():
    return "Hello, World!"
```

This is the final paragraph that concludes the sample text. It provides a natural ending to the document and helps test edge cases in chunking algorithms."""


@pytest.fixture
def simple_config():
    """Fixture providing a basic ChunkingConfig for simple chunking."""
    from ingestion.chunker.config import ChunkingConfig
    return ChunkingConfig(
        chunk_size=100,
        chunk_overlap=20,
        max_chunk_size=200,
        min_chunk_size=10,
        use_semantic_splitting=False,
        preserve_structure=True
    )


@pytest.fixture
def semantic_config():
    """Fixture providing a ChunkingConfig for semantic chunking."""
    from ingestion.chunker.config import ChunkingConfig
    return ChunkingConfig(
        chunk_size=200,
        chunk_overlap=50,
        max_chunk_size=400,
        min_chunk_size=20,
        use_semantic_splitting=True,
        preserve_structure=True
    )


@pytest.fixture
def document_metadata():
    """Fixture providing sample document metadata."""
    return {
        "title": "Test Document",
        "source": "test.txt",
        "author": "Test Author",
        "category": "testing",
        "tags": ["test", "chunking", "rag"],
        "version": "1.0",
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def mock_embedding_client():
    """Fixture providing a mock embedding client."""
    from unittest.mock import Mock
    mock_client = Mock()
    mock_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )
    return mock_client


@pytest.fixture
def mock_llm_model():
    """Fixture providing a mock LLM model."""
    from unittest.mock import Mock
    mock_model = Mock()
    return mock_model


@pytest.fixture
def long_text():
    """Fixture providing long text for testing chunking algorithms."""
    base_paragraph = "This is a test paragraph that will be repeated multiple times to create a long document. " * 10
    return "\n\n".join([f"Section {i}: {base_paragraph}" for i in range(10)])


@pytest.fixture
def structured_markdown():
    """Fixture providing structured markdown content for testing."""
    return """# Main Title

## Introduction

This is the introduction section with some basic information about the topic.

## Background

### Historical Context

Here we provide historical background information that is quite detailed.

### Current State

This section describes the current state of affairs with multiple subsections.

#### Subsection A

Content for subsection A with specific details.

#### Subsection B

Content for subsection B with different details.

## Methodology

1. First step in the methodology
2. Second step with detailed explanation
3. Third step that concludes the process

### Data Collection

- Primary data sources
- Secondary data sources
- Data validation methods

### Analysis Approach

The analysis approach involves several key components:

```python
def analyze_data(data):
    # Perform analysis
    results = process(data)
    return results
```

## Results

The results show significant findings across multiple dimensions.

### Quantitative Results

| Metric | Value | Significance |
|--------|--------|-------------|
| Accuracy | 95% | High |
| Precision | 92% | High |
| Recall | 88% | Medium |

### Qualitative Results

The qualitative analysis revealed several important themes:

- Theme 1: Importance of data quality
- Theme 2: Need for robust validation
- Theme 3: Significance of context

## Conclusion

This concludes our analysis with key takeaways and future recommendations."""


@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Auto-used fixture to mock external dependencies in tests."""
    import pytest
    from unittest.mock import patch, Mock
    
    with patch('ingestion.chunker.semantic.get_embedding_client') as mock_embedding, \
         patch('ingestion.chunker.semantic.get_ingestion_model') as mock_model, \
         patch('ingestion.chunker.simple.get_embedding_client') as mock_embedding2, \
         patch('ingestion.chunker.simple.get_ingestion_model') as mock_model2:
        
        # Configure mocks
        mock_embedding.return_value = Mock()
        mock_model.return_value = Mock()
        mock_embedding2.return_value = Mock()
        mock_model2.return_value = Mock()
        
        yield {
            'embedding_client': mock_embedding,
            'ingestion_model': mock_model,
            'embedding_client2': mock_embedding2,
            'ingestion_model2': mock_model2
        }


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "async: mark test as requiring async support")


# Helper functions for tests
class TestHelpers:
    """Helper class with utility methods for tests."""
    
    @staticmethod
    def assert_valid_chunk(chunk, config=None):
        """Assert that a chunk is valid according to configuration."""
        from ingestion.chunker.chunk import DocumentChunk
        
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(chunk.content, str)
        assert isinstance(chunk.index, int)
        assert isinstance(chunk.start_char, int)
        assert isinstance(chunk.end_char, int)
        assert isinstance(chunk.metadata, dict)
        assert chunk.index >= 0
        assert chunk.start_char >= 0
        assert chunk.end_char >= chunk.start_char
        assert len(chunk.content.strip()) > 0
        
        if config:
            content_length = len(chunk.content.strip())
            assert content_length >= config.min_chunk_size
            assert content_length <= config.max_chunk_size
    
    @staticmethod
    def assert_valid_chunk_sequence(chunks, config=None):
        """Assert that a sequence of chunks is valid."""
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        for i, chunk in enumerate(chunks):
            TestHelpers.assert_valid_chunk(chunk, config)
            assert chunk.index == i
            
            # Check metadata consistency
            if i > 0:
                assert chunk.metadata.get("total_chunks") == chunks[0].metadata.get("total_chunks")
                assert chunk.metadata.get("title") == chunks[0].metadata.get("title")
                assert chunk.metadata.get("source") == chunks[0].metadata.get("source")


@pytest.fixture
def test_helpers():
    """Fixture providing test helper utilities."""
    return TestHelpers