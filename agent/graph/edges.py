from pydantic import BaseModel, Field
from datetime import date
from typing import Optional


class Dating(BaseModel):
    start_date: date = Field(..., description="When the dating began")
    end_date: Optional[date] = Field(None, description="When it ended")
    status: Optional[str]    = Field(None, description="e.g. 'on', 'off'")


class FeudWith(BaseModel):
    start_date: date        = Field(..., description="When the feud began")
    public_intensity: str   = Field(..., description="low|medium|high")


class CollaboratedOn(BaseModel):
    project_name: str       = Field(..., description="Name of film/song/show")
    year: int               = Field(..., description="Production year")
    role: Optional[str]     = Field(None, description="e.g. 'director', 'co-star'")


class RumoredWith(BaseModel):
    rumor_text: str         = Field(..., description="Text of the circulating rumor")
    first_seen_source: str  = Field(..., description="Origin (e.g. TMZ, People.com)")
    confidence_score: float = Field(..., description="0.0–1.0 confidence")


edge_types = {
    "Dating": Dating,
    "FeudWith": FeudWith,
    "CollaboratedOn": CollaboratedOn,
    "RumoredWith": RumoredWith,
}

edge_type_map = {
    ("Person", "Person"): ["Dating", "FeudWith", "CollaboratedOn", "RumoredWith"],
}