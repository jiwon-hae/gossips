"""
Configuration for news data collection pipeline.
"""

import os

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum
from dataclasses import dataclass, field

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))   

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


@dataclass
class NewsSourceConfig:
    """Configuration for news sources."""
    name: str
    base_url: str
    rss_feeds: List[str]
    rate_limit_seconds: float = 1.0
    max_articles_per_request: int = 100
    enabled: bool = True


@dataclass
class EventClassificationConfig:
    """Configuration for event classification."""
    categories: List[EventCategory]
    keywords_per_category: Dict[EventCategory, List[str]]
    confidence_threshold: float = 0.7
    use_llm_classification: bool = True
    fallback_to_keyword_matching: bool = True


@dataclass
class DataCollectionConfig:
    """Main configuration for data collection pipeline."""
    # Collection settings
    collection_interval_hours: int = 6
    max_articles_per_run: int = 500
    days_to_collect: int = 30

    # Storage settings
    output_directory: str = f"{project_root}/data/collected_news"
    database_path: str = f"{project_root}/data/news_database.db"
    export_formats: List[str] = None

    # Processing settings
    clean_html: bool = True
    remove_duplicates: bool = True
    min_headline_length: int = 10
    max_headline_length: int = 200

    # Classification settings
    classification: EventClassificationConfig = None

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["json", "csv", "parquet"]

        if self.classification is None:
            self.classification = EventClassificationConfig(
                categories=list(EventCategory),
                keywords_per_category=self._get_default_keywords(),
                confidence_threshold=0.7,
                use_llm_classification=True,
                fallback_to_keyword_matching=True
            )

    def _get_default_keywords(self) -> Dict[EventCategory, List[str]]:
        """Get default keywords for each celebrity event category."""
        return {
            # Relationship Events
            EventCategory.DIVORCE: [
                "divorce", "divorcing", "separated", "separation", "split up", "end marriage",
                "custody", "alimony", "prenup", "settlement", "ex-husband", "ex-wife",
                "filing for divorce", "divorce papers", "irreconcilable differences"
            ],
            EventCategory.BREAKUP: [
                "breakup", "break up", "broke up", "split", "ended relationship",
                "called it quits", "no longer together", "relationship over",
                "ex-boyfriend", "ex-girlfriend", "former couple", "parted ways"
            ],
            EventCategory.ENGAGEMENT: [
                "engaged", "engagement", "proposal", "proposed", "ring", "fiancé", "fiancée",
                "wedding planning", "getting married", "said yes", "popped the question"
            ],
            EventCategory.MARRIAGE: [
                "married", "wedding", "ceremony", "tied the knot", "exchanged vows",
                "newlyweds", "husband", "wife", "matrimony", "nuptials", "altar"
            ],
            EventCategory.DATING: [
                "dating", "relationship", "boyfriend", "girlfriend", "romantic", "couple",
                "together", "seeing each other", "new romance", "spotted together",
                "holding hands", "kissing", "love interest", "rumored relationship"
            ],
            EventCategory.CHEATING: [
                "cheating", "affair", "unfaithful", "infidelity", "two-timing", "betrayed",
                "caught cheating", "secret relationship", "love triangle", "mistress"
            ],
            EventCategory.RECONCILIATION: [
                "back together", "reconciled", "reunion", "rekindled", "second chance",
                "working things out", "got back", "reunited", "patched things up"
            ],

            # Conflict Events
            EventCategory.FEUD: [
                "feud", "rivalry", "enemies", "bad blood", "tension", "animosity",
                "ongoing dispute", "long-standing conflict", "bitter rivalry"
            ],
            EventCategory.FIGHT: [
                "fight", "argument", "confrontation", "altercation", "heated exchange",
                "verbal fight", "public spat", "shouting match", "disagreement"
            ],
            EventCategory.LAWSUIT: [
                "lawsuit", "suing", "sued", "legal action", "court case", "litigation",
                "defamation", "damages", "settlement", "trial", "lawyer", "attorney"
            ],
            EventCategory.CONTROVERSY: [
                "controversy", "controversial", "backlash", "criticism", "outrage",
                "scandal", "problematic", "under fire", "facing criticism"
            ],
            EventCategory.SCANDAL: [
                "scandal", "shocking", "exposed", "leaked", "revelation", "bombshell",
                "damaging", "embarrassing", "career-ending", "reputation"
            ],
            EventCategory.BEEF: [
                "beef", "diss track", "rivalry", "calling out", "shade", "throwing shade",
                "subliminal", "responding", "clap back", "feud"
            ],
            EventCategory.DISS: [
                "diss", "diss track", "calling out", "insulted", "roasted", "burned",
                "savage", "clap back", "response track", "shade"
            ],

            # Personal Events
            EventCategory.PREGNANCY: [
                "pregnant", "pregnancy", "expecting", "baby on the way", "due",
                "maternity", "baby bump", "first child", "second child", "twins"
            ],
            EventCategory.BIRTH: [
                "gave birth", "born", "newborn", "baby", "delivered", "welcomed",
                "new arrival", "labor", "hospital", "healthy baby"
            ],
            EventCategory.DEATH: [
                "died", "death", "passed away", "funeral", "memorial", "tribute",
                "obituary", "mourning", "loss", "tragic", "sudden death"
            ],
            EventCategory.HEALTH_ISSUE: [
                "health", "illness", "sick", "hospital", "surgery", "medical",
                "diagnosis", "treatment", "recovery", "health scare", "condition"
            ],
            EventCategory.ADDICTION: [
                "addiction", "substance abuse", "drug problem", "drinking problem",
                "overdose", "relapse", "sobriety", "clean", "intervention"
            ],
            EventCategory.REHAB: [
                "rehab", "rehabilitation", "treatment center", "getting help",
                "seeking treatment", "recovery", "sober", "facility"
            ],
            EventCategory.MENTAL_HEALTH: [
                "depression", "anxiety", "mental health", "therapy", "counseling",
                "breakdown", "struggling", "wellness", "self-care", "healing"
            ],

            # Career Events
            EventCategory.NEW_PROJECT: [
                "new movie", "new album", "new show", "upcoming", "filming",
                "recording", "project", "role", "cast", "signed", "deal"
            ],
            EventCategory.COLLABORATION: [
                "collaboration", "featuring", "duet", "partnership", "working with",
                "team up", "joint project", "featuring", "guest appearance"
            ],
            EventCategory.AWARD: [
                "won", "award", "trophy", "prize", "honored", "recognition",
                "achievement", "victory", "winner", "champion"
            ],
            EventCategory.NOMINATION: [
                "nominated", "nomination", "shortlisted", "contender", "up for",
                "in the running", "candidate", "eligible"
            ],
            EventCategory.RETIREMENT: [
                "retirement", "retiring", "stepping down", "final", "farewell",
                "last performance", "ending career", "calling it quits"
            ],
            EventCategory.COMEBACK: [
                "comeback", "return", "back", "revival", "resurgence",
                "making a comeback", "triumphant return", "back in action"
            ],

            # Social Events
            EventCategory.PARTY: [
                "party", "celebration", "birthday party", "event", "gala",
                "gathering", "bash", "soirée", "exclusive party", "VIP"
            ],
            EventCategory.RED_CARPET: [
                "red carpet", "premiere", "awards show", "fashion", "outfit",
                "dressed", "stunning", "glamorous", "style", "look"
            ],
            EventCategory.VACATION: [
                "vacation", "holiday", "getaway", "trip", "traveling", "exotic",
                "beach", "resort", "relaxing", "time off"
            ],
            EventCategory.FRIENDSHIP: [
                "friendship", "best friends", "close friends", "squad", "besties",
                "supportive", "loyal", "tight-knit", "inner circle"
            ],
            EventCategory.FAMILY_DRAMA: [
                "family drama", "family feud", "estranged", "family problems",
                "sibling rivalry", "parent issues", "family conflict"
            ],

            # Social Media Events
            EventCategory.SOCIAL_MEDIA_DRAMA: [
                "twitter", "instagram", "social media", "posted", "deleted",
                "viral", "trending", "hashtag", "online drama", "internet"
            ],
            EventCategory.VIRAL_MOMENT: [
                "viral", "trending", "internet sensation", "meme", "went viral",
                "breaking the internet", "social media buzz", "online phenomenon"
            ],
            EventCategory.APOLOGY: [
                "apology", "apologized", "sorry", "regret", "mistake",
                "taking responsibility", "making amends", "public apology"
            ],
            EventCategory.STATEMENT: [
                "statement", "announced", "revealed", "confirmed", "denied",
                "clarified", "addressed", "spoke out", "breaking silence"
            ],

            # Fashion & Style
            EventCategory.FASHION_MOMENT: [
                "fashion", "outfit", "dress", "style", "designer", "couture",
                "fashion week", "trendsetter", "iconic look", "fashion statement"
            ],
            EventCategory.STYLE_CHANGE: [
                "makeover", "new look", "style change", "transformation",
                "haircut", "hair color", "fashion evolution", "reinvention"
            ],

            # Financial Events
            EventCategory.BUSINESS_VENTURE: [
                "business", "company", "brand", "launching", "entrepreneur",
                "investment", "startup", "venture", "business deal"
            ],
            EventCategory.FINANCIAL_TROUBLE: [
                "bankruptcy", "financial trouble", "debt", "money problems",
                "lawsuit", "taxes", "financial crisis", "broke"
            ],
            EventCategory.CHARITY: [
                "charity", "donation", "philanthropic", "giving back", "foundation",
                "fundraising", "humanitarian", "cause", "helping"
            ],
            EventCategory.ENDORSEMENT: [
                "endorsement", "spokesperson", "ambassador", "deal", "contract",
                "partnership", "brand", "sponsorship", "campaign"
            ]
        }

DEFAULT_CONFIG = DataCollectionConfig(
    collection_interval_hours=6,
    max_articles_per_run=500,
    days_to_collect=30,
    output_directory=f"{project_root}/data/collected_news",
    database_path=f"{project_root}/data/news_database.db"
)
