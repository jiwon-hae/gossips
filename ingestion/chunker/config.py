from dataclasses import dataclass

@dataclass
class ChunkingConfig:
    """
    Configuration class for document chunking parameters.
    
    This class defines the parameters used for splitting documents into chunks
    for processing in the RAG pipeline.
    
    Attributes:
        chunk_size (int): Target size for each chunk in characters. Default: 1000
        chunk_overlap (int): Number of overlapping characters between chunks. Default: 200
        max_chunk_size (int): Maximum allowed chunk size. Default: 2000
        min_chunk_size (int): Minimum allowed chunk size. Default: 100
        use_semantic_splitting (bool): Whether to use semantic boundaries. Default: True
        preserve_structure (bool): Whether to preserve document structure. Default: True
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    use_semantic_splitting: bool = True
    preserve_structure: bool = True

    def __post_init__(self):
        """
        Validate the chunking configuration to ensure logical consistency.
        
        Raises:
            ValueError: If chunk_overlap >= chunk_size or min_chunk_size <= 0
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")
