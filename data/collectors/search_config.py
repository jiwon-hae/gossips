from typing import Optional, List
from dataclasses import dataclass

# try:
#     from .category import EventCategory
# except ImportError:
#     import sys
#     import os
#     sys.path.append(os.path.dirname(
#         os.path.dirname(os.path.abspath(__file__))))
    
#     from category import EventCategory

@dataclass
class Period:
    HOURLY = '1h'
    DAILY = "25h"
    WEEKLY = "7d"
    MONTHLY = "1m"
    YEARLY = "1y"
    
from enum import Enum

class EventCategory(Enum):
    """Event categories for celebrity news classification."""
    # Relationship Events
    DIVORCE = "divorce"
    BREAKUP = "breakup"
    ENGAGEMENT = "engagement"
    MARRIAGE = "marriage"
    DATING = "dating"
    CHEATING = "cheating"
    RECONCILIATION = "reconciliation"

    # Conflict Events
    FEUD = "feud"
    FIGHT = "fight"
    LAWSUIT = "lawsuit"
    CONTROVERSY = "controversy"
    SCANDAL = "scandal"
    BEEF = "beef"
    DISS = "diss"

    # Personal Events
    PREGNANCY = "pregnancy"
    BIRTH = "birth"
    DEATH = "death"
    HEALTH_ISSUE = "health_issue"
    ADDICTION = "addiction"
    REHAB = "rehab"
    MENTAL_HEALTH = "mental_health"

    # Career Events
    NEW_PROJECT = "new_project"
    COLLABORATION = "collaboration"
    AWARD = "award"
    NOMINATION = "nomination"
    RETIREMENT = "retirement"
    COMEBACK = "comeback"
    CAREER_MILESTONE = "career_milestone"

    # Social Events
    PARTY = "party"
    RED_CARPET = "red_carpet"
    VACATION = "vacation"
    FRIENDSHIP = "friendship"
    FAMILY_DRAMA = "family_drama"

    # Financial Events
    BUSINESS_VENTURE = "business_venture"
    FINANCIAL_TROUBLE = "financial_trouble"
    CHARITY = "charity"
    ENDORSEMENT = "endorsement"

    # Social Media Events
    SOCIAL_MEDIA_DRAMA = "social_media_drama"
    VIRAL_MOMENT = "viral_moment"
    APOLOGY = "apology"
    STATEMENT = "statement"

    # Fashion & Style
    FASHION_MOMENT = "fashion_moment"
    STYLE_CHANGE = "style_change"

    # Other
    OTHER = "other"