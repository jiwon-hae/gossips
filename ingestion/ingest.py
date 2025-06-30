import os
import asyncio
import logging
import json
import glob
import argparse

from pydantic import Field, field_validator
try:
    from .graph.graph_builder import create_graph_builder
    from .chunker.config import ChunkingConfig
    from .chunker.chunker import create_chunker
    
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    
    from ingestion.graph.graph_builder import create_graph_builder
    from ingestion.chunker.config import ChunkingConfig
    from ingestion.chunker.chunker import create_chunker
    

logger = logging.setLogger(__name__)


class IngestionConfig:
    """
    Configuration class for the document ingestion pipeline.
    
    This class serves as a placeholder for ingestion configuration parameters.
    Currently empty, but can be extended with specific configuration options
    such as supported file types, processing settings, etc.
    """
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=10000)
    use_semantic_chunking: bool = True
    extract_entities: bool = True
    skip_graph_building: bool = Field(
        default=False, description="Skip knowledge graph building for faster ingestion")

    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError(
                f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v


class DocumentIngestionPipeline:
    """
    Main pipeline for ingesting documents into the RAG system.
    
    This class orchestrates the document ingestion process, including
    loading documents, processing them through the chunking pipeline,
    and storing the results for retrieval.
    
    Attributes:
        config (IngestinConfig): Configuration for the ingestion process
        document_folder (str): Path to the folder containing documents to ingest
        clean_before_ingest (bool): Whether to clean existing data before ingestion
    """

    def __init__(
        self,
        config: IngestionConfig,
        document_folder: str = 'documents',
        clean_before_ingest: bool = False
    ):
        """
        Initialize the document ingestion pipeline.
        
        Args:
            config (IngestinConfig): Configuration for the ingestion process
            document_folder (str): Path to the folder containing documents. Default: 'documents'
            clean_before_ingest (bool): Whether to clean existing data before ingestion. Default: False
        """
        self.config = config
        self.document_folder = document_folder
        self.clean_before_ingest = clean_before_ingest

        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking
        )

        self.chunker = create_chunker(self.chunker_config)
        #TODO(jiwon-hae) : Implement embedder
        self.embedder = craete_embedder()
        self.graph_builder = create_graph_builder()