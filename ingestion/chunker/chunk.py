from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """
    Represents a chunk of a document with associated metadata.
    
    This class stores information about a specific chunk of text extracted
    from a larger document, including its position and metadata.
    
    Attributes:
        content (str): The actual text content of the chunk
        index (int): Sequential index of this chunk within the document
        start_char (int): Starting character position in the original document
        end_char (int): Ending character position in the original document
        metadata (Dict[str, Any]): Additional metadata associated with the chunk
        token_count (Optional[int]): Number of tokens in the chunk, if calculated
    """
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None