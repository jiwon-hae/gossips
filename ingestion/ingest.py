import logging
import asyncio
import argparse

from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    from .graph.graph_builder import create_graph_builder
    from .chunker.config import ChunkingConfig
    from .chunker.chunker import create_chunker
    from .embed.embedder import create_embedder
    from ..agent.graph.graph import initialize_graph
    from ..vector_store.postgresql_store import *
    from .config import *
    from .file_utils import *
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    from agent.graph.graph import initialize_graph, close_graph
    from ingestion.chunker.config import ChunkingConfig
    from ingestion.chunker.chunker import create_chunker
    from ingestion.file_utils import *
    from ingestion.embed.embedder import create_embedder
    from ingestion.graph.graph_builder import create_graph_builder
    from ingestion.config import *
    from vector_store.postgresql_store import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class IngestionResult(BaseModel):
    """Result of document ingestion."""
    document_id: str
    title: str
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    processing_time_ms: float
    errors: List[str] = Field(default_factory=list)
    

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
        documents_folder: str = 'documents',
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
        self.document_folder = documents_folder
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

        documents = find_documents(document_folder=self.document_folder, doc_patterns = [".doc", ".txt", ".pdf"])
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

        # Document chunking
        chunks = await self.chunker.chunk_document(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata
        )

        if not chunks:
            logger.warning(f"No chunks created for {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=(
                    datetime.now() - start_time).total_seconds() * 1000,
                errors=["No chunks created"]
            )

        logging.info(f"Created {len(chunks)} chunks")

        # Extract entities if configured
        entities_extracted = 0
        if self.config.extract_entities:
            chunks = await self.graph_builder.extract_entities_from_chunks(chunks)
            entities_extracted = sum(
                len(chunk.metadata.get("entities", {})).get("celebrities", []) +
                len(chunk.metadata.get("entities", {})).get("events", [])
                for chunk in chunks
            )

            logger.info(f"Extracted {entities_extracted} entities")

        # Generate embeddings
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        logger.info(f"Genearted embeddings for {len(embedded_chunks)} chunks")

        document_id = await save_to_postgres(
            document_title,
            document_source,
            document_content,
            embedded_chunks,
            document_metadata
        )

        logger.info(f"Saved document to PostgreSQL with ID: {document_id}")

        # Add to knowledge graph (if enabled)
        relationships_created = 0
        graph_errors = []

        if not self.config.skip_graph_building:
            try:
                logger.info(
                    "Building knowledge graph relationships (this may take several minutes)...")
                graph_result = await self.graph_builder.add_document_to_graph(
                    chunks=embedded_chunks,
                    document_title=document_title,
                    document_source=document_source,
                    document_metadata=document_metadata
                )

                relationships_created = graph_result.get("episodes_created", 0)
                graph_errors = graph_result.get("errors", [])

                logger.info(
                    f"Added {relationships_created} episodes to knowledge graph")

            except Exception as e:
                error_msg = f"Failed to add to knowledge graph: {str(e)}"
                logger.error(error_msg)
                graph_errors.append(error_msg)
        else:
            logger.info(
                "Skipping knowledge graph building (skip_graph_building=True)")

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(chunks),
            entities_extracted=entities_extracted,
            relationships_created=relationships_created,
            processing_time_ms=processing_time,
            errors=graph_errors
        )

    async def _clean_database(self):
        await clean_databases()

        # Clean knowledge graph
        await self.graph_builder.clear_graph()
        logger.info("Cleaned knowledge graph")


async def main():
    """Main function for running ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into vector DB and knowledge graph")
    parser.add_argument("--documents", "-d",
                        default="documents", help="Documents folder path")
    parser.add_argument("--clean", "-c", action="store_true",
                        help="Clean existing data before ingestion")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Chunk size for splitting documents")
    parser.add_argument("--chunk-overlap", type=int,
                        default=200, help="Chunk overlap size")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Disable semantic chunking")
    parser.add_argument("--no-entities", action="store_true",
                        help="Disable entity extraction")
    parser.add_argument("--fast", "-f", action="store_true",
                        help="Fast mode: skip knowledge graph building")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create ingestion configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic,
        extract_entities=not args.no_entities,
        skip_graph_building=args.fast
    )
    
    # Create and run pipeline
    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=args.clean
    )
    
    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total} documents processed")
    
    try:
        start_time = datetime.now()
        
        results = await pipeline.ingest_documents(progress_callback)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Print summary
        print("\n" + "="*50)
        print("INGESTION SUMMARY")
        print("="*50)
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Total entities extracted: {sum(r.entities_extracted for r in results)}")
        print(f"Total graph episodes: {sum(r.relationships_created for r in results)}")
        print(f"Total errors: {sum(len(r.errors) for r in results)}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print()
        
        # Print individual results
        for result in results:
            status = "✓" if not result.errors else "✗"
            print(f"{status} {result.title}: {result.chunks_created} chunks, {result.entities_extracted} entities")
            
            if result.errors:
                for error in result.errors:
                    print(f"  Error: {error}")
        
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
