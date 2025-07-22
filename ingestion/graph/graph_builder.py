import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from typing import Optional, Any, List, Dict
from ingestion.chunker.chunk import DocumentChunk
from graphiti_core import Graphiti
from agent.graph.client import GraphitiClient
from datetime import datetime, timezone

import spacy
import wikipedia
import logging
import asyncio


logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


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
        self.graph_client = GraphitiClient()
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
                await self.client.add_episode(
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
                error_msg = f"Failed to add chunk {chunk.index} to graph: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

                # Continue processing other chunks even if one fails
                continue

        result = {
            "episodes_created": episodes_created,
            "total_chunks": len(chunks),
            "errors": errors
        }

        logger.info(
            f"Graph building complete: {episodes_created} episodes created, {len(errors)} errors")
        return result

    def _prepare_episode_content(
        self,
        chunk: DocumentChunk,
        document_title: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prepare episode content with minimal context to avoid token limits.
        
        Args:
            chunk: Document chunk
            document_title: Title of the document
            document_metadata: Additional metadata
        
        Returns:
            Formatted episode content (optimized for Graphiti)
        """
        # Limit chunk content to avoid Graphiti's 8192 token limit
        # Estimate ~4 chars per token, keep content under 6000 chars to leave room for processing
        max_content_length = 6000

        content = chunk.content
        if len(content) > max_content_length:
            # Truncate content but try to end at a sentence boundary
            truncated = content[:max_content_length]
            last_sentence_end = max(
                truncated.rfind('. '),
                truncated.rfind('! '),
                truncated.rfind('? ')
            )

            if last_sentence_end > max_content_length * 0.7:  # If we can keep 70% and end cleanly
                content = truncated[:last_sentence_end + 1] + " [TRUNCATED]"
            else:
                content = truncated + "... [TRUNCATED]"

            logger.warning(
                f"Truncated chunk {chunk.index} from {len(chunk.content)} to {len(content)} chars for Graphiti")

        # Add minimal context (just document title for now)
        if document_title and len(content) < max_content_length - 100:
            episode_content = f"[Doc: {document_title[:50]}]\n\n{content}"
        else:
            episode_content = content

        return episode_content

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (4 chars per token)."""
        return len(text) // 4

    def _is_content_too_large(self, content: str, max_tokens: int = 7000) -> bool:
        """Check if content is too large for Graphiti processing."""
        return self._estimate_tokens(content) > max_tokens

    def _is_celebrity(self, name):
        try:
            summary = wikipedia.summary(name, sentences=1)
            return any(word in summary.lower() for word in ["singer", "actor", "actress", "athlete", "rapper", "celebrity", "player"])
        except:
            return False

    def _extract_celebrities(self, text: str) -> List[str]:
        found_celebrities = set()
        doc = nlp(text)
        for entity in doc.ents:
            if entity.label_ == "PERSON" and self._is_celebrity(entity.text):
                found_celebrities.add(entity.text)

        return list(found_celebrities)


    async def extract_entities_from_chunks(
        self,
        chunks: List[DocumentChunk],
        extract_celebrities: bool = True,
        extract_events: bool = True,
    ) -> List[DocumentChunk]:
        """
        Extract entities from chunks and add to metadata.
        
        Args:
            chunks: List of document chunks
            extract_companies: Whether to extract company names
            extract_technologies: Whether to extract technology terms
            extract_people: Whether to extract person names
        
        Returns:
            Chunks with entity metadata added
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks")

        enriched_chunks = []

        for chunk in chunks:
            content = chunk.content
            if extract_celebrities:
                extracted_celebs = self._extract_celebrities(content)
                existing = set(chunk.metadata.get('celebrities', []))
                chunk.metadata['celebrities'] = list(existing.union(extracted_celebs))
            
            enriched_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "entity_extraction_date": datetime.now().isoformat()
                },
                token_count=chunk.token_count
            )

            if hasattr(chunk, 'embedding'):
                enriched_chunk.embedding = chunk.embedding

            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks

    async def clear_graph(self):
        """Clear all data from the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        
        logger.warning("Clearing knowledge graph...")
        await self.graph_client.clear_graph()
        logger.info("Knowledge graph cleared")
    
    async def close(self):
        """Close graph client."""
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False
    

# Factory function
def create_graph_builder() -> GraphBuilder:
    """Create graph builder instance."""
    return GraphBuilder()