
from typing import Optional, Any, List, Dict
from ingestion.chunker.chunk import DocumentChunk
from graphiti_core import Graphiti
from datetime import datetime, timezone

import logging
import asyncio

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builder class for constructing knowledge graphs from document chunks.
    
    This class uses the Graphiti library to build and manage knowledge graphs
    that represent relationships and entities found in ingested documents.
    
    Attributes:
        client: GraphitiClient instance for interacting with the graph database
    """

    def __init__(self):
        """
        Initialize the GraphBuilder with a GraphitiClient instance.
        
        Note: This appears to reference GraphitiClient but imports Graphiti.
        The implementation may need to be updated to match the actual API.
        """
        self.client = GraphitiClient()
        self._initialized = False

    async def initialize(self):
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True

    async def add_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        document_title, document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add document chunks to the knowledge graph.
        
        Args:
            chunks: List of document chunks
            document_title: Title of the document
            document_source: Source of the document
            document_metadata: Additional metadata
            batch_size: Number of chunks to process in each batch
        
        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()

        if not chunks:
            return {"episodes_crated": 0, "errors": []}

        logger.info(
            f"Adding {len(chunks)} chunks to knowledge graph for document: {document_title}")
        logger.info(
            "⚠️ Large chunks will be truncated to avoid Graphiti token limits.")

        # Check for oversized chunks and warn
        oversized_chunks = [i for i, chunk in enumerate(
            chunks) if len(chunk.content) > 6000]
        if oversized_chunks:
            logger.warning(
                f"Found {len(oversized_chunks)} chunks over 6000 chars that will be truncated: {oversized_chunks}")

        episodes_created = 0
        errors = []

        for i, chunk in enumerate(chunks):
            try:
                # Create episode ID
                episode_id = f"{document_source}_{chunk.index}_{datetime.now().timestamp()}"

                # Prepare episode content with size limits
                episode_content = self._prepare_episode_content(
                    chunk,
                    document_title,
                    document_metadata
                )
                # Create source description (shorter)
                source_description = f"Document: {document_title} (Chunk: {chunk.index})"

                # Add episode to graph
                await self.graph_client.add_episode(
                    episode_id=episode_id,
                    content=episode_content,
                    source=source_description,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "document_title": document_title,
                        "document_source": document_source,
                        "chunk_index": chunk.index,
                        "original_length": len(chunk.content),
                        "processed_length": len(episode_content)
                    }
                )

                episodes_created += 1
                logger.info(
                    f"✓ Added episode {episode_id} to knowledge graph ({episodes_created}/{len(chunks)})")

                # Small delay between each episode to reduce API pressure
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)

            except Exception as e:
                pass
