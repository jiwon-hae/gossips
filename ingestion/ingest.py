import logging
import json
import glob
import argparse

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import Field, field_validator

try:
    from .ingestion_result import IngestionResult
    from .graph.graph_builder import create_graph_builder
    from .chunker.config import ChunkingConfig
    from .chunker.chunker import create_chunker
    from .embed.embedder import create_embedder
    from ..agent.graph.graph import initialize_graph
    from ..vector_store.postgresql_store import initilize_db
    from .document import *
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    from agent.graph.graph import initialize_graph, close_graph
    from ingestion.chunker.config import ChunkingConfig
    from ingestion.chunker.chunker import create_chunker
    from ingestion.document import *
    from ingestion.graph.graph_builder import create_graph_builder
    from ingestion.ingestion_result import IngestionResult
    from vector_store.postgresql_store import initilize_db, close_db


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
        self.embedder = create_embedder()
        self.graph_builder = create_graph_builder()
        self._initialized = False

    async def initialize(self):
        """Initilize database connection"""

        if self._initialized:
            return

        logger.info("Initializing ingestion pipeline")

        await initilize_db()
        await initialize_graph()
        await self.graph_builder.initialize()

        self._initialized = True
        logger.info("Ingestion pipeline initialize")

    async def close(self):
        if self._initialized:
            await self.graph_builder.close()
            await close_graph()
            await close_db()
            self._initialized = False

    async def ingest_documents(
        self,
        progress_callback: Optional[callable] = None
    ) -> List[IngestionResult]:
        if not self._initialized:
            await self.initialize()

        if self.clean_before_ingest:
            await self._clean_database()

        documents = find_documents(document_folder = self.document_folder)
        if not documents:
            logger.warning(f"No douments found in {self.document_folder}")
            return []

        logger.info(f"Found {len(documents)} documents to process")
        results = []

        for idx, file_path in enumerate(documents):
            try:
                logger.info(
                    f"Processing file {idx + 1}/{len(documents)}: {file_path}")
                result = await self._ingest_single_document(file_path)
                results.append(result)
                if progress_callback:
                    progress_callback(idx + 1, len(documents))
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
    

    async def _ingest_single_document(self, document_path: str) -> IngestionResult:
        """
        Ingest a single document
        
        Args:
            document_path: Path to the document file
        
        Resurns:
            Ingestion result
        """
        start_time = datetime.now()
        document_content = read_document(document_path)
        document_title = extract_title(
            content=document_content, file_path=document_path)
        document_source = os.path.relpath(document_path, self.documents_folder)
        document_metadata = extract_document_metadata(
            content=document_content, file_path=document_path)
        
        logging.info(f"Processing document: {document_title}")
        

    async def _clean_database():
        raise NotImplementedError("clean database not yet implemented")
