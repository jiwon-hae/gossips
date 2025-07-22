from enum import Enum
from typing import Dict

class EventCategory(Enum):
    RELATIONSHIP = 'relationship'
    CONFLICT = 'conflict'
    PERSONAL = 'personal'
    CAREER = 'career'
    SOCIAL = 'social'
    FINANCIAL = 'financial'
    SOCIAL_MEDIA = 'social_media'
    FASHION = 'fashion'
    OTHERS = 'others'

class Event(Enum):
    # Relationship Events
    DIVORCE         = 'divorce'
    BREAKUP         = 'breakup'
    ENGAGEMENT      = 'engagement'
    MARRIAGE        = 'marriage'
    DATING          = 'dating'
    CHEATING        = 'cheating'
    RECONCILIATION  = 'reconciliation'

    # Conflict Events
    FEUD            = 'feud'
    FIGHT           = 'fight'
    LAWSUIT         = 'lawsuit'
    CONTROVERSY     = 'controversy'
    SCANDAL         = 'scandal'
    BEEF            = 'beef'
    DISS            = 'diss'
    ACCUSATION      = 'accusation'

    # Personal Events
    PREGNANCY       = 'pregnancy'
    BIRTH           = 'birth'
    DEATH           = 'death'
    HEALTH_ISSUE    = 'health_issue'
    ADDICTION       = 'addiction'
    REHAB           = 'rehab'
    MENTAL_HEALTH   = 'mental_health'
    FAMILY          = 'family'

    # Career Events
    NEW_PROJECT       = 'new_project'
    COLLABORATION     = 'collaboration'
    AWARD             = 'award'
    NOMINATION        = 'nomination'
    RETIREMENT        = 'retirement'
    COMEBACK          = 'comeback'
    CAREER_MILESTONE  = 'career_milestone'

    # Social Events
    PARTY             = 'party'
    RED_CARPET        = 'red_carpet'
    VACATION          = 'vacation'
    FRIENDSHIP        = 'friendship'
    FAMILY_DRAMA      = 'family_drama'

    # Financial Events
    BUSINESS_VENTURE   = 'business_venture'
    FINANCIAL_TROUBLE  = 'financial_trouble'
    CHARITY            = 'charity'
    ENDORSEMENT        = 'endorsement'

    # Social Media Events
    SOCIAL_MEDIA_DRAMA = 'social_media_drama'
    VIRAL_MOMENT       = 'viral_moment'
    APOLOGY            = 'apology'
    STATEMENT          = 'statement'

    # Fashion & Style
    FASHION_MOMENT     = 'fashion_moment'
    STYLE_CHANGE       = 'style_change'

    # Catch-all
    OTHER              = 'other'
    
    @classmethod
    def values(cls):
        return [e.value for e in cls]


# Mapping each fine-grained Event → its high-level EventCategory
EVENT_TO_CATEGORY: Dict[Event, EventCategory] = {
    # Relationship
    Event.DIVORCE:        EventCategory.RELATIONSHIP,
    Event.BREAKUP:        EventCategory.RELATIONSHIP,
    Event.ENGAGEMENT:     EventCategory.RELATIONSHIP,
    Event.MARRIAGE:       EventCategory.RELATIONSHIP,
    Event.DATING:         EventCategory.RELATIONSHIP,
    Event.CHEATING:       EventCategory.RELATIONSHIP,
    Event.RECONCILIATION: EventCategory.RELATIONSHIP,

    # Conflict
    Event.FEUD:        EventCategory.CONFLICT,
    Event.FIGHT:       EventCategory.CONFLICT,
    Event.LAWSUIT:     EventCategory.CONFLICT,
    Event.CONTROVERSY: EventCategory.CONFLICT,
    Event.SCANDAL:     EventCategory.CONFLICT,
    Event.BEEF:        EventCategory.CONFLICT,
    Event.DISS:        EventCategory.CONFLICT,
    Event.ACCUSATION:  EventCategory.CONFLICT,

    # Personal
    Event.PREGNANCY:      EventCategory.PERSONAL,
    Event.BIRTH:          EventCategory.PERSONAL,
    Event.DEATH:          EventCategory.PERSONAL,
    Event.HEALTH_ISSUE:   EventCategory.PERSONAL,
    Event.ADDICTION:      EventCategory.PERSONAL,
    Event.REHAB:          EventCategory.PERSONAL,
    Event.MENTAL_HEALTH:  EventCategory.PERSONAL,
    Event.FAMILY:    EventCategory.PERSONAL,

    # Career
    Event.NEW_PROJECT:      EventCategory.CAREER,
    Event.COLLABORATION:    EventCategory.CAREER,
    Event.AWARD:            EventCategory.CAREER,
    Event.NOMINATION:       EventCategory.CAREER,
    Event.RETIREMENT:       EventCategory.CAREER,
    Event.COMEBACK:         EventCategory.CAREER,
    Event.CAREER_MILESTONE: EventCategory.CAREER,

    # Social
    Event.PARTY:        EventCategory.SOCIAL,
    Event.RED_CARPET:   EventCategory.SOCIAL,
    Event.VACATION:     EventCategory.SOCIAL,
    Event.FRIENDSHIP:   EventCategory.SOCIAL,
    Event.FAMILY_DRAMA: EventCategory.SOCIAL,

    # Financial
    Event.BUSINESS_VENTURE:  EventCategory.FINANCIAL,
    Event.FINANCIAL_TROUBLE: EventCategory.FINANCIAL,
    Event.CHARITY:           EventCategory.FINANCIAL,
    Event.ENDORSEMENT:       EventCategory.FINANCIAL,

    # Social Media
    Event.SOCIAL_MEDIA_DRAMA: EventCategory.SOCIAL_MEDIA,
    Event.VIRAL_MOMENT:       EventCategory.SOCIAL_MEDIA,
    Event.APOLOGY:            EventCategory.SOCIAL_MEDIA,
    Event.STATEMENT:          EventCategory.SOCIAL_MEDIA,

    # Fashion
    Event.FASHION_MOMENT: EventCategory.FASHION,
    Event.STYLE_CHANGE:   EventCategory.FASHION,

    # Catch-all
    Event.OTHER:          EventCategory.OTHERS,
    
}