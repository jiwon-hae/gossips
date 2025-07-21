from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from datetime import date


class RelationshipStatus(Enum):
    MARRIED = 'married'
    DIVORCED = 'divorced'
    

@dataclass
class Relationship:
    partner : str
    relationship : RelationshipStatus
    start_yr : date
    end_yr : Optional[date] = None

@dataclass
class PersonalInfo:
    name: str
    occupation: str
    current_relationship : Optional[Relationship] = None
    past_relationships : Optional[List[Relationship]] = None
    parents: List[str] = field(default_factory=list)