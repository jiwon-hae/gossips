"""
Data models for news collection pipeline.
"""
import os
import sys
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)



from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from data.collection.config import EventCategory


@dataclass
class NewsArticle:
    """Data model for a news article."""
    title: str
    url: str
    source: str
    published_date: datetime
    summary: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    language: str = "en"
    
    # Collection metadata
    collected_at: datetime = field(default_factory=datetime.now)
    collection_source: str = "google_news"
    
    # Processing metadata
    processed: bool = False
    cleaned: bool = False
    classified: bool = False
    
    # Classification results
    predicted_category: Optional[EventCategory] = None
    classification_confidence: Optional[float] = None
    classification_method: Optional[str] = None
    keywords_matched: List[str] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure URL is valid
        if not self.url.startswith(('http://', 'https://')):
            self.url = f"https://{self.url}"
    
    @property
    def id(self) -> str:
        """Generate a unique ID for the article."""
        import hashlib
        content = f"{self.title}_{self.url}_{self.published_date.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_date": self.published_date.isoformat(),
            "summary": self.summary,
            "content": self.content,
            "author": self.author,
            "category": self.category,
            "language": self.language,
            "collected_at": self.collected_at.isoformat(),
            "collection_source": self.collection_source,
            "processed": self.processed,
            "cleaned": self.cleaned,
            "classified": self.classified,
            "predicted_category": self.predicted_category.value if self.predicted_category else None,
            "classification_confidence": self.classification_confidence,
            "classification_method": self.classification_method,
            "keywords_matched": self.keywords_matched,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsArticle':
        """Create article from dictionary."""
        # Parse datetime fields
        published_date = datetime.fromisoformat(data["published_date"].replace('Z', '+00:00'))
        collected_at = datetime.fromisoformat(data["collected_at"].replace('Z', '+00:00'))
        
        # Parse category if exists
        predicted_category = None
        if data.get("predicted_category"):
            predicted_category = EventCategory(data["predicted_category"])
        
        return cls(
            title=data["title"],
            url=data["url"],
            source=data["source"],
            published_date=published_date,
            summary=data.get("summary"),
            content=data.get("content"),
            author=data.get("author"),
            category=data.get("category"),
            language=data.get("language", "en"),
            collected_at=collected_at,
            collection_source=data.get("collection_source", "google_news"),
            processed=data.get("processed", False),
            cleaned=data.get("cleaned", False),
            classified=data.get("classified", False),
            predicted_category=predicted_category,
            classification_confidence=data.get("classification_confidence"),
            classification_method=data.get("classification_method"),
            keywords_matched=data.get("keywords_matched", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class ClassificationResult:
    """Result of event classification."""
    category: EventCategory
    confidence: float
    method: str  # "llm", "keyword", "hybrid"
    keywords_matched: List[str] = field(default_factory=list)
    raw_scores: Dict[EventCategory, float] = field(default_factory=dict)
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "method": self.method,
            "keywords_matched": self.keywords_matched,
            "raw_scores": {k.value: v for k, v in self.raw_scores.items()},
            "explanation": self.explanation
        }


@dataclass
class CollectionStats:
    """Statistics about collection run."""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_articles_found: int = 0
    articles_collected: int = 0
    articles_classified: int = 0
    articles_skipped: int = 0
    errors_encountered: int = 0
    sources_processed: List[str] = field(default_factory=list)
    categories_distribution: Dict[EventCategory, int] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def articles_per_second(self) -> Optional[float]:
        """Calculate articles processed per second."""
        duration = self.duration_seconds
        if duration and duration > 0:
            return self.articles_collected / duration
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "total_articles_found": self.total_articles_found,
            "articles_collected": self.articles_collected,
            "articles_classified": self.articles_classified,
            "articles_skipped": self.articles_skipped,
            "errors_encountered": self.errors_encountered,
            "articles_per_second": self.articles_per_second,
            "sources_processed": self.sources_processed,
            "categories_distribution": {k.value: v for k, v in self.categories_distribution.items()}
        }


@dataclass
class TrainingDataSample:
    """Sample for training event classifier."""
    text: str  # Usually the headline
    label: EventCategory
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "text": self.text,
            "label": self.label.value,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata
        }