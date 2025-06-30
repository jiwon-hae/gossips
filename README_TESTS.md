# Chunker Tests Documentation

This document describes the comprehensive test suite for the chunker module in the RAG system.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and configuration
└── ingestion/
    └── chunker/
        ├── __init__.py
        ├── test_config.py         # ChunkingConfig tests
        ├── test_chunk.py          # DocumentChunk tests  
        ├── test_base_chunker.py   # BaseChunker abstract class tests
        ├── test_simple_chunker.py # SimpleChunker implementation tests
        ├── test_semantic_chunker.py # SemanticChunker implementation tests
        ├── test_factory.py        # create_chunker factory tests
        └── test_integration.py    # Integration tests
```

## Test Coverage

### ChunkingConfig (`test_config.py`)
- **Default values**: Validates all default configuration values
- **Custom values**: Tests initialization with custom parameters
- **Validation logic**: Tests validation rules for overlap and min_chunk_size
- **Edge cases**: Tests boundary conditions and error cases
- **Dataclass behavior**: Tests equality, inequality, and string representation

### DocumentChunk (`test_chunk.py`)
- **Basic initialization**: Tests creation with all required fields
- **Optional fields**: Tests token_count and complex metadata
- **Equality comparisons**: Tests dataclass equality behavior
- **Edge cases**: Tests empty content and complex metadata structures

### BaseChunker (`test_base_chunker.py`)
- **Abstract nature**: Verifies cannot be instantiated directly
- **Concrete implementation**: Tests with a concrete test implementation
- **Helper methods**: Tests validation and filtering utilities
- **Integration flow**: Tests complete chunking workflow

### SimpleChunker (`test_simple_chunker.py`)
- **Initialization**: Tests proper setup and configuration
- **Empty content handling**: Tests edge cases with empty/whitespace content
- **Single vs multiple chunks**: Tests content that fits in one vs multiple chunks
- **Paragraph splitting**: Tests content splitting on paragraph boundaries
- **Metadata preservation**: Tests custom metadata handling
- **Positioning**: Tests chunk position calculation
- **Overlap behavior**: Tests chunk overlap functionality
- **Long content**: Tests handling of very long documents

### SemanticChunker (`test_semantic_chunker.py`)
- **Initialization**: Tests setup with mocked dependencies
- **Semantic splitting logic**: Tests LLM-based chunking (mocked)
- **Fallback behavior**: Tests fallback to simple chunking on errors
- **Structural splitting**: Tests markdown and structured content parsing
- **LLM integration**: Tests integration with language models (mocked)
- **Error handling**: Tests graceful degradation when LLM fails

### Factory Function (`test_factory.py`)
- **Chunker selection**: Tests creation of appropriate chunker type
- **Configuration preservation**: Tests that config is properly passed
- **Type consistency**: Tests that both chunkers implement same interface
- **Multiple instances**: Tests creating multiple independent chunkers

### Integration Tests (`test_integration.py`)
- **End-to-end workflows**: Tests complete chunking processes
- **Cross-component interaction**: Tests how components work together
- **Large document processing**: Tests performance with large content
- **Concurrent usage**: Tests thread safety
- **Configuration combinations**: Tests various config scenarios

## Running Tests

### Basic Usage

```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --test-type unit
python run_tests.py --test-type integration

# Run with coverage
python run_tests.py --coverage --html-coverage

# Run specific test file
python run_tests.py --specific tests/ingestion/chunker/test_config.py

# Skip slow tests
python run_tests.py --fast
```

### Using pytest directly

```bash
# Run all chunker tests
pytest tests/ingestion/chunker/ -v

# Run with coverage
pytest tests/ingestion/chunker/ --cov=ingestion.chunker --cov-report=html

# Run specific test class
pytest tests/ingestion/chunker/test_config.py::TestChunkingConfig -v

# Run async tests only
pytest tests/ingestion/chunker/ -m async

# Run integration tests only  
pytest tests/ingestion/chunker/ -m integration
```

## Test Fixtures

The `conftest.py` file provides shared fixtures:

- **`sample_text`**: Realistic sample text for testing
- **`simple_config`/`semantic_config`**: Pre-configured ChunkingConfig instances
- **`document_metadata`**: Sample metadata for testing
- **`long_text`**: Long content for performance testing
- **`structured_markdown`**: Structured content for testing
- **`test_helpers`**: Utility functions for test assertions
- **`mock_external_dependencies`**: Auto-mocked external dependencies

## Test Markers

Tests are marked with pytest markers for easy filtering:

- **`@pytest.mark.unit`**: Unit tests
- **`@pytest.mark.integration`**: Integration tests  
- **`@pytest.mark.async`**: Tests requiring async support
- **`@pytest.mark.slow`**: Slow-running tests (performance tests)

## Mocking Strategy

### External Dependencies
- **LLM models**: Mocked to avoid API calls during testing
- **Embedding clients**: Mocked for consistent test behavior
- **Prompt loaders**: Mocked to control LLM prompts

### Internal Dependencies
- **File I/O**: Tests use in-memory content
- **Network calls**: All external calls are mocked
- **Time-dependent operations**: Mocked for deterministic tests

## Test Data

Tests use realistic data scenarios:

- **Short documents**: Single paragraph content
- **Medium documents**: Multi-paragraph with structure
- **Long documents**: Large content requiring multiple chunks
- **Structured content**: Markdown with headers, lists, tables, code blocks
- **Edge cases**: Empty content, whitespace-only, single characters

## Assertions and Validation

### Custom Assertions
The `TestHelpers` class provides:
- **`assert_valid_chunk()`**: Validates individual chunks
- **`assert_valid_chunk_sequence()`**: Validates chunk sequences
- **Configuration compliance**: Ensures chunks meet config requirements

### Standard Assertions
- **Type checking**: Ensures correct return types
- **Boundary validation**: Tests min/max constraints
- **Content integrity**: Verifies content preservation
- **Metadata consistency**: Ensures metadata propagation

## Continuous Integration

### GitHub Actions (if using)
```yaml
- name: Run Chunker Tests
  run: |
    pip install pytest pytest-asyncio pytest-cov
    python run_tests.py --coverage
```

### Local Development
```bash
# Quick test run during development
python run_tests.py --fast

# Full test suite before commit
python run_tests.py --coverage --html-coverage
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project root is in Python path
2. **Async Test Failures**: Check pytest-asyncio is installed
3. **Mock Failures**: Verify external dependencies are properly mocked
4. **Slow Tests**: Use `--fast` flag to skip performance tests

### Debug Mode
```bash
# Run with maximum verbosity
pytest tests/ingestion/chunker/ -vvv --tb=long

# Run specific failing test
pytest tests/ingestion/chunker/test_config.py::TestChunkingConfig::test_validation_overlap_greater_than_chunk_size -vvv
```

## Coverage Goals

Target coverage metrics:
- **Overall**: > 90%
- **Unit tests**: > 95% 
- **Integration tests**: > 80%
- **Critical paths**: 100% (validation, error handling)

## Contributing

When adding new tests:

1. **Follow naming conventions**: `test_*.py` files, `Test*` classes, `test_*` methods
2. **Use appropriate markers**: Mark tests with `@pytest.mark.unit` or `@pytest.mark.integration`
3. **Mock external dependencies**: Don't make real API calls in tests
4. **Add docstrings**: Document what each test verifies
5. **Use fixtures**: Leverage shared fixtures from `conftest.py`
6. **Test edge cases**: Include boundary conditions and error scenarios