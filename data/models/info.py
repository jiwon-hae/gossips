from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from datetime import date


class RelationshipType(Enum):
    MARRIAGE = 'marriage'
    DIVORCE = 'divorced'
    

@dataclass
class Relationship:
    partner : str
    relationship : RelationshipType
    start_yr : date
    end_yr : Optional[date] = None

@dataclass
class PersonalInfo:
    name: str
    occupation: str
    spouse: List[Relationship] = field(default_factory=list)