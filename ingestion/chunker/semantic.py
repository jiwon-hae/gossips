from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from agent.providers import *
from utils.prompt import PromptLoader

import logging
import re

from ingestion.chunker.chunk import DocumentChunk
from ingestion.chunker.config import ChunkingConfig
from ingestion.chunker.chunker import BaseChunker

embedding_client = get_embedding_client()
ingestion_model = get_ingestion_model()
_prompt_loader = PromptLoader(path='ingestion')
logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """Semantic document chunker using LLM for intelligent splitting."""

    def __init__(self, config: ChunkingConfig):
        """
        Initialize chunker.
        
        Args:
            config: Chunking configuration
        """
        super().__init__(config)
        self.client = embedding_client
        self.model = ingestion_model

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk a document into semantically coherent pieces.
        
        Args:
            content: Document content
            title: Document title
            source: Document source
            metadata: Additional metadata
        
        Returns:
            List of document chunks
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            **(metadata or {})
        }

        if self.config.use_semantic_splitting and len(content) > self.config.chunk_size:
            try:
                semantic_chunks = await self._semantic_chunk(content)
                if semantic_chunks:
                    return self._create_chunk(
                        semantic_chunks,
                        content,
                        base_metadata
                    )
            except Exception as e:
                logger.warning(
                    f"Semantic chunking failed, falling back to simple chunking: {e}")

    async def _semantic_chunk(self, content: str) -> List[str]:
        """
        Perform semantic chunking using LLM.
        
        Args:
            content: Content to chunk
        
        Returns:
            List of chunk boundaries
        """
        # First, split on natural boundaries
        sections = self._split_on_structure(content)

        # Group sections into semantic chunks
        chunks = []
        current_chunk = ""

        for section in sections:
            # Check if adding this section would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + section if current_chunk else section

            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                # Current chunk is ready, decide if we should split the section
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Handle oversized sections
                if len(section) > self.config.max_chunk_size:
                    # Split the section semantically
                    sub_chunks = await self._split_long_section(section)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = section

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.strip()) >= self.config.min_chunk_size]

    def _split_on_structure(self, content: str) -> List[str]:
        """
        Split content on structural boundaries.
        
        Args:
            content: Content to split
        
        Returns:
            List of sections
        """
        # Split on markdown headers, paragraphs, and other structural elements
        patterns = [
            r'\n#{1,6}\s+.+?\n',    # Markdown headers
            r'\n\n+',               # Multiple newlines (paragraph breaks)
            r'\n[-*+]\s+',          # List items
            r'\n\d+\.\s+',          # Numbered lists
            r'\n```.*?```\n',       # Code blocks
            r'\n\|\s*.+?\|\s*\n',   # Tables
        ]

        # Split by patterns but keep the separators
        sections = [content]

        for pattern in patterns:
            new_sections = []
            for section in sections:
                parts = re.split(f'({pattern})', section,
                                 flags=re.MULTILINE | re.DOTALL)
                new_sections.extend([part for part in parts if part.strip()])
            sections = new_sections

        return sections

    async def _split_long_section(self, section: str) -> List[str]:
        """
        Split a long section using LLM for semantic boundaries.
        
        Args:
            section: Section to split
        
        Returns:
            List of sub-chunks
        """
        try:
            prompt = _prompt_loader.render(
                'semantic.j2', section=section, chunk_size=self.config.chunk_size, max_chunk_size=self.config.max_chunk_size)

            # Use Pydantic AI for LLM calls
            from pydantic_ai import Agent
            temp_agent = Agent(self.model)

            response = await temp_agent.run(prompt)
            result = response.ouptut
            chunks = [chunk.strip() for chunk in result.split("---CHUNK---")]

            # Validate chunks
            valid_chunks = []
            for chunk in chunks:
                if (self.config.min_chunk_size <= len(chunk) <= self.config.max_chunk_size):
                    valid_chunks.append(chunk)

            return valid_chunks if valid_chunks else self._simple_split(section)

        except Exception as e:
            logger.error(f"LLM chunking failed: {e}")
            return self._simple_split(section)

    def _simple_split(self, text: str) -> List[str]:
        """
        Simple text splitting as fallback.
        
        Args:
            text: Text to split
        
        Returns:
            List of chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.config.chunk_size

            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break

            # Try to end at a sentence boundary
            chunk_end = end
            for i in range(end, max(start + self.config.min_chunk_size, end - 200), -1):
                if text[i] in '.!?\n':
                    chunk_end = i + 1
                    break

            chunks.append(text[start:chunk_end])
            start = chunk_end - self.config.chunk_overlap

        return chunks

    def _simple_chunk(
        self, content: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Simple rule-based chunking.
        
        Args:
            content: Content to chunk
            base_metadata: Base metadata for chunks
        
        Returns:
            List of document chunks
        """
        chunks = self._simple_split(content)
        return self._create_chunk(chunks, content, base_metadata)

    def _create_chunk(
        self,
        chunks: List[str],
        original_content: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Create DocumentChunk objects from text chunks.
        
        Args:
            chunks: List of chunk texts
            original_content: Original document content
            base_metadata: Base metadata
        
        Returns:
            List of DocumentChunk objects
        """
        chunk_objects = []
        current_pos = 0

        for i, chunk_text in enumerate(chunks):
            # Find the position of this chunk in the original content
            start_pos = original_content.find(chunk_text, current_pos)
            if start_pos == -1:
                # Fallback: estimate position
                start_pos = current_pos

            end_pos = start_pos + len(chunk_text)

            # Create chunk metadata
            chunk_metadata = {
                **base_metadata,
                "chunk_method": "semantic" if self.config.use_semantic_splitting else "simple",
                "total_chunks": len(chunks)
            }

            chunk_objects.append(DocumentChunk(
                content=chunk_text.strip(),
                index=i,
                start_char=start_pos,
                end_char=end_pos,
                metadata=chunk_metadata
            ))

            current_pos = end_pos

        return chunk_objects
