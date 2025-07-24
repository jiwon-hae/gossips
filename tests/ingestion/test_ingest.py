import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Mock external dependencies at module level
import sys
from unittest.mock import Mock

# Mock missing modules
missing_modules = [
    'openai',
    'pydantic_ai',
    'pydantic_ai.models.openai', 
    'pydantic_ai.providers.openai',
    'jinja2',
    'pandas',
    'psycopg2',
    'asyncpg'
]

for module_name in missing_modules:
    if module_name not in sys.modules:
        sys.modules[module_name] = Mock()

try:
    from ingestion.ingest import DocumentIngestionPipeline
    from ingestion.config import IngestionConfig
except ImportError:
    import os
    import sys
    
    # Add the project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, project_root)
    
    from ingestion.ingest import DocumentIngestionPipeline
    from ingestion.config import IngestionConfig


class TestIngestArticle:
    """Test suite for the _ingest_article method in DocumentIngestionPipeline."""
    
    @pytest.fixture
    def mock_config(self):
        """Fixture providing a mock IngestionConfig."""
        return IngestionConfig(
            chunk_size=1000,
            chunk_overlap=200,
            max_chunk_size=2000,
            use_semantic_chunking=False,
            extract_entities=True,
            skip_graph_building=False
        )
    
    @pytest.fixture
    def pipeline(self, mock_config):
        """Fixture providing a DocumentIngestionPipeline instance."""
        with patch('ingestion.ingest.create_chunker'), \
             patch('ingestion.ingest.create_embedder'), \
             patch('ingestion.ingest.create_graph_builder'):
            pipeline = DocumentIngestionPipeline(config=mock_config)
            return pipeline
    
    @pytest.fixture
    def sample_article_data(self):
        """Fixture providing sample article data."""
        return {
            "title": "Justin Bieber Releases New Album",
            "id": "abc123",
            "content": "Justin Bieber has released a new album titled 'Changes'. The album features collaborations with various artists...",
            "keywords": ["music", "album", "pop"],
            "url": "https://example.com/justin-bieber-new-album",
            "celeb": "Justin Bieber",
            "publisher": "Music News",
            "published_date": "2024-01-15"
        }
    
    @pytest.fixture
    def sample_profile_data(self):
        """Fixture providing sample celebrity profile data."""
        return {
            "name": "Justin Bieber",
            "occupation": "['singer', 'songwriter']",
            "spouse": "[{'partner': 'Hailey Bieber', 'relationship': 'married', 'start_yr': '2018-09-13', 'end_yr': None}]",
            "parents": "[]"
        }
    
    @pytest.fixture
    def temp_article_file(self, sample_article_data):
        """Fixture creating a temporary article JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(sample_article_data, f, ensure_ascii=False, indent=2)
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_ingest_article_success(self, pipeline, temp_article_file, sample_article_data, sample_profile_data):
        """Test successful article ingestion."""
        with patch.object(pipeline, '_get_profiles', return_value=sample_profile_data):
            result = pipeline._ingest_article(temp_article_file)
            
            # Assert the result contains both article data and profile data
            assert result is not None
            assert result["title"] == sample_article_data["title"]
            assert result["id"] == sample_article_data["id"]
            assert result["content"] == sample_article_data["content"]
            assert result["celeb"] == sample_article_data["celeb"]
            
            # Assert profile data is merged
            assert result["name"] == sample_profile_data["name"]
            assert result["occupation"] == sample_profile_data["occupation"]
            assert result["spouse"] == sample_profile_data["spouse"]
    
    def test_ingest_article_with_none_profile(self, pipeline, temp_article_file, sample_article_data):
        """Test article ingestion when profile is not found."""
        with patch.object(pipeline, '_get_profiles', return_value=None):
            result = pipeline._ingest_article(temp_article_file)
            
            # Should still return article data even without profile
            assert result is not None
            assert result["title"] == sample_article_data["title"]
            assert result["celeb"] == sample_article_data["celeb"]
            
            # Profile data should not be present
            assert "name" not in result or result.get("name") is None
            assert "occupation" not in result or result.get("occupation") is None
    
    def test_ingest_article_file_not_found(self, pipeline):
        """Test handling of non-existent file."""
        non_existent_path = Path("/non/existent/file.json")
        
        with patch('ingestion.ingest.logger') as mock_logger:
            result = pipeline._ingest_article(non_existent_path)
            
            # Should return None for non-existent file
            assert result is None
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "does not exists" in error_call
    
    def test_ingest_article_invalid_json(self, pipeline):
        """Test handling of invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            f.write("invalid json content {")
            temp_path = f.name
        
        try:
            with patch('ingestion.ingest.logger') as mock_logger:
                result = pipeline._ingest_article(Path(temp_path))
                
                # Should return None for invalid JSON
                assert result is None
                mock_logger.error.assert_called_once()
                error_call = mock_logger.error.call_args[0][0]
                assert "Failed to ingest article" in error_call
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_ingest_article_missing_celeb_field(self, pipeline):
        """Test handling of article data missing 'celeb' field."""
        article_data_no_celeb = {
            "title": "Some Article",
            "content": "Some content",
            "id": "xyz789"
            # Missing 'celeb' field
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(article_data_no_celeb, f)
            temp_path = f.name
        
        try:
            with patch.object(pipeline, '_get_profiles') as mock_get_profiles, \
                 patch('ingestion.ingest.logger') as mock_logger:
                
                result = pipeline._ingest_article(Path(temp_path))
                
                # Should handle KeyError gracefully
                assert result is None
                mock_logger.error.assert_called_once()
                mock_get_profiles.assert_not_called()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_ingest_article_profile_exception(self, pipeline, temp_article_file, sample_article_data):
        """Test handling when _get_profiles raises an exception."""
        with patch.object(pipeline, '_get_profiles', side_effect=Exception("Profile error")), \
             patch('ingestion.ingest.logger') as mock_logger:
            
            result = pipeline._ingest_article(temp_article_file)
            
            # Should handle exception gracefully
            assert result is None
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "Failed to ingest article" in error_call
    
    def test_ingest_article_encoding_handling(self, pipeline, sample_profile_data):
        """Test proper handling of UTF-8 encoding in article files."""
        article_with_unicode = {
            "title": "Célébrité News: Émilie's New Rôle",
            "celeb": "Émilie Björk",
            "content": "Article with special characters: café, naïve, résumé",
            "id": "unicode123"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(article_with_unicode, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            with patch.object(pipeline, '_get_profiles', return_value=sample_profile_data):
                result = pipeline._ingest_article(Path(temp_path))
                
                # Should properly handle Unicode characters
                assert result is not None
                assert result["title"] == article_with_unicode["title"]
                assert result["celeb"] == article_with_unicode["celeb"]
                assert result["content"] == article_with_unicode["content"]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_ingest_article_empty_profile_merge(self, pipeline, temp_article_file, sample_article_data):
        """Test merging with empty profile data."""
        empty_profile = {}
        
        with patch.object(pipeline, '_get_profiles', return_value=empty_profile):
            result = pipeline._ingest_article(temp_article_file)
            
            # Should merge successfully even with empty profile
            assert result is not None
            assert result["title"] == sample_article_data["title"]
            assert result["celeb"] == sample_article_data["celeb"]
            # Original article data should be preserved
            for key, value in sample_article_data.items():
                assert result[key] == value
    
    def test_ingest_article_profile_override(self, pipeline, temp_article_file):
        """Test that profile data correctly overrides article data for same keys."""
        # Create article with overlapping keys
        article_data = {
            "title": "Article Title",
            "celeb": "Test Celebrity",
            "name": "Article Name",  # This should be overridden by profile
            "content": "Article content"
        }
        
        profile_data = {
            "name": "Profile Name",  # This should override article name
            "occupation": "Actor"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(article_data, f)
            temp_path = f.name
        
        try:
            with patch.object(pipeline, '_get_profiles', return_value=profile_data):
                result = pipeline._ingest_article(Path(temp_path))
                
                assert result is not None
                assert result["name"] == "Profile Name"  # Profile should override
                assert result["occupation"] == "Actor"  # Profile data should be added
                assert result["title"] == "Article Title"  # Article data should remain
                assert result["content"] == "Article content"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIngestArticleIntegration:
    """Integration tests for _ingest_article method."""
    
    @pytest.fixture
    def mock_config(self):
        return IngestionConfig(
            chunk_size=500,
            chunk_overlap=100,
            max_chunk_size=1000,
            use_semantic_chunking=False,
            extract_entities=False,
            skip_graph_building=True
        )
    
    # @pytest.fixture
    # def pipeline_with_mocked_deps(self, mock_config):
    #     """Pipeline with all external dependencies mocked."""
    #     with patch('ingestion.ingest.create_chunker'), \
    #          patch('ingestion.ingest.create_embedder'), \
    #          patch('ingestion.ingest.create_graph_builder'), \
    #          patch('ingestion.ingest.pd') as mock_pd:
            
    #         # Mock pandas DataFrame
    #         mock_df = Mock()
    #         mock_df.empty = False
    #         mock_df.iloc = Mock()
    #         mock_df.iloc.__getitem__ = Mock()
    #         mock_df.iloc.__getitem__.return_value.str.lower.return_value.__eq__ = Mock(return_value=Mock())
            
    #         mock_pd.read_csv.return_value = mock_df
            
    #         pipeline = DocumentIngestionPipeline(config=mock_config)
    #         return pipeline
    
    # @pytest.mark.integration  
    # def test_ingest_article_full_flow(self, pipeline_with_mocked_deps):
    #     """Test the complete flow of article ingestion."""
    #     article_data = {
    #         "title": "Integration Test Article",
    #         "celeb": "Test Star",
    #         "content": "This is a complete integration test for article ingestion.",
    #         "id": "integration123",
    #         "url": "https://test.com/article",
    #         "keywords": ["test", "integration"]
    #     }
        
    #     with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
    #         json.dump(article_data, f)
    #         temp_path = f.name
        
    #     try:
    #         # Mock the profile data
    #         with patch.object(pipeline_with_mocked_deps, '_get_profiles', return_value={"name": "Test Star", "occupation": "Actor"}):
    #             result = pipeline_with_mocked_deps._ingest_article(Path(temp_path))
                
    #             # Verify complete result
    #             assert result is not None
    #             assert isinstance(result, dict)
    #             assert "title" in result
    #             assert "content" in result
    #             assert "name" in result
    #             assert "occupation" in result
    #     finally:
    #         if os.path.exists(temp_path):
    #             os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])