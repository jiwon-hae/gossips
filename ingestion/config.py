from pydantic import Field, field_validator, BaseModel
from pydantic import BaseModel, Field

class IngestionConfig(BaseModel):
    """
    Configuration class for the document ingestion pipeline.
    
    This class serves as a placeholder for ingestion configuration parameters.
    Currently empty, but can be extended with specific configuration options
    such as supported file types, processing settings, etc.
    """
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=10000)
    extract_entities: bool = True
    extract_sentiment : bool = True
    extract_events : bool = True
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