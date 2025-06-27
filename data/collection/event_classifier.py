"""
Celebrity event classifier for categorizing news headlines.
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime
from collections import Counter

from data.collection.models import NewsArticle, ClassificationResult, TrainingDataSample
from data.collection.config import EventCategory, DataCollectionConfig

logger = logging.getLogger(__name__)


class CelebrityEventClassifier:
    """Classifier for celebrity events based on headlines and content."""
    
    def __init__(self, config: DataCollectionConfig = None):
        """Initialize the classifier."""
        self.config = config
        self.keywords_per_category = config.classification.keywords_per_category if config else {}
        
        # Enhanced keyword patterns for better matching
        self.compiled_patterns = self._compile_keyword_patterns()
        
        # Relationship indicators for better context
        self.relationship_indicators = {
            'romantic': ['dating', 'boyfriend', 'girlfriend', 'couple', 'romance', 'love', 'together'],
            'marriage': ['husband', 'wife', 'married', 'spouse', 'wedding', 'ceremony'],
            'family': ['mother', 'father', 'sister', 'brother', 'daughter', 'son', 'family'],
            'professional': ['co-star', 'colleague', 'director', 'producer', 'manager']
        }
        
        # Event severity/impact indicators
        self.severity_indicators = {
            'high': ['shocking', 'devastating', 'explosive', 'bombshell', 'tragic', 'scandal'],
            'medium': ['surprising', 'unexpected', 'dramatic', 'heated', 'tense'],
            'low': ['minor', 'small', 'brief', 'quiet', 'private']
        }
    
    def _compile_keyword_patterns(self) -> Dict[EventCategory, List[re.Pattern]]:
        """Compile regex patterns for each category."""
        compiled = {}
        
        for category, keywords in self.keywords_per_category.items():
            patterns = []
            for keyword in keywords:
                # Create flexible regex pattern
                # Handle multi-word phrases
                if ' ' in keyword:
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                else:
                    # Single word - allow for variations
                    pattern = r'\b' + re.escape(keyword) + r's?\b'  # Handle plurals
                
                patterns.append(re.compile(pattern, re.IGNORECASE))
            
            compiled[category] = patterns
        
        return compiled
    
    def _extract_celebrity_names(self, text: str) -> List[str]:
        """Extract potential celebrity names from text."""
        # Simple name extraction - looks for capitalized words
        # In production, you'd want a more sophisticated NER model
        words = text.split()
        potential_names = []
        
        i = 0
        while i < len(words):
            word = re.sub(r'[^\w]', '', words[i])
            if word.istitle() and len(word) > 2:
                # Check if next word is also capitalized (full name)
                if i + 1 < len(words):
                    next_word = re.sub(r'[^\w]', '', words[i + 1])
                    if next_word.istitle() and len(next_word) > 2:
                        potential_names.append(f"{word} {next_word}")
                        i += 2
                        continue
                
                potential_names.append(word)
            i += 1
        
        return potential_names
    
    def _keyword_based_classification(self, text: str) -> Dict[EventCategory, float]:
        """Classify text based on keyword matching."""
        scores = {}
        text_lower = text.lower()
        
        for category, patterns in self.compiled_patterns.items():
            score = 0.0
            matches = []
            
            for pattern in patterns:
                pattern_matches = pattern.findall(text)
                if pattern_matches:
                    # Weight matches based on position and frequency
                    for match in pattern_matches:
                        # Higher score for matches in title vs content
                        position_weight = 1.0
                        if text.find(match) < len(text) * 0.3:  # First 30% of text
                            position_weight = 1.5
                        
                        score += position_weight
                        matches.append(match.lower())
            
            # Normalize score by text length and number of keywords
            if score > 0:
                score = score / (len(text.split()) / 100)  # Normalize by text length
                scores[category] = min(score, 1.0)  # Cap at 1.0
        
        return scores
    
    def _relationship_context_boost(self, text: str, scores: Dict[EventCategory, float]) -> Dict[EventCategory, float]:
        """Boost scores based on relationship context."""
        text_lower = text.lower()
        
        # Detect relationship type
        relationship_type = None
        for rel_type, indicators in self.relationship_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                relationship_type = rel_type
                break
        
        if relationship_type:
            # Boost relevant categories based on relationship context
            if relationship_type in ['romantic', 'marriage']:
                boost_categories = [
                    EventCategory.DIVORCE, EventCategory.BREAKUP, EventCategory.ENGAGEMENT,
                    EventCategory.MARRIAGE, EventCategory.DATING, EventCategory.CHEATING,
                    EventCategory.RECONCILIATION
                ]
                for category in boost_categories:
                    if category in scores:
                        scores[category] *= 1.3
        
        return scores
    
    def _content_quality_filter(self, text: str) -> bool:
        """Filter out low-quality or irrelevant content."""
        text_lower = text.lower()
        
        # Filter out promotional content
        promo_indicators = ['buy now', 'subscribe', 'click here', 'advertisement', 'sponsored']
        if any(indicator in text_lower for indicator in promo_indicators):
            return False
        
        # Filter out very short headlines
        if len(text.split()) < 4:
            return False
        
        # Must contain at least one celebrity-related term
        celebrity_terms = ['celebrity', 'star', 'actor', 'actress', 'singer', 'musician', 'artist']
        celebrity_names = self._extract_celebrity_names(text)
        
        if not any(term in text_lower for term in celebrity_terms) and not celebrity_names:
            return False
        
        return True
    
    async def classify_article(self, article: NewsArticle) -> ClassificationResult:
        """Classify a single article."""
        # Combine title and summary for classification
        text_to_classify = article.title
        if article.summary:
            text_to_classify += " " + article.summary
        
        # Quality filter
        if not self._content_quality_filter(text_to_classify):
            return ClassificationResult(
                category=EventCategory.OTHER,
                confidence=0.0,
                method="filtered_out",
                explanation="Article filtered out due to low quality or irrelevance"
            )
        
        # Keyword-based classification
        scores = self._keyword_based_classification(text_to_classify)
        
        # Apply context boosts
        scores = self._relationship_context_boost(text_to_classify, scores)
        
        # Find best category
        if not scores:
            return ClassificationResult(
                category=EventCategory.OTHER,
                confidence=0.0,
                method="keyword",
                explanation="No matching keywords found"
            )
        
        best_category = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_category]
        
        # Extract matched keywords for this category
        matched_keywords = []
        for pattern in self.compiled_patterns.get(best_category, []):
            matches = pattern.findall(text_to_classify)
            matched_keywords.extend([m.lower() for m in matches])
        
        return ClassificationResult(
            category=best_category,
            confidence=confidence,
            method="keyword",
            keywords_matched=list(set(matched_keywords)),
            raw_scores=scores,
            explanation=f"Matched {len(matched_keywords)} keywords for {best_category.value}"
        )
    
    async def classify_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Classify a list of articles."""
        classified_articles = []
        
        for article in articles:
            try:
                result = await self.classify_article(article)
                
                # Update article with classification results
                article.predicted_category = result.category
                article.classification_confidence = result.confidence
                article.classification_method = result.method
                article.keywords_matched = result.keywords_matched
                article.classified = True
                
                # Add classification metadata
                article.metadata['classification_result'] = result.to_dict()
                
                classified_articles.append(article)
                
            except Exception as e:
                logger.error(f"Error classifying article '{article.title}': {str(e)}")
                article.classified = False
                classified_articles.append(article)
        
        logger.info(f"Classified {len(classified_articles)} articles")
        return classified_articles
    
    def generate_training_data(self, articles: List[NewsArticle], 
                             min_confidence: float = 0.5) -> List[TrainingDataSample]:
        """Generate training data samples from classified articles."""
        training_samples = []
        
        for article in articles:
            if (article.classified and 
                article.classification_confidence and 
                article.classification_confidence >= min_confidence and
                article.predicted_category != EventCategory.OTHER):
                
                sample = TrainingDataSample(
                    text=article.title,
                    label=article.predicted_category,
                    source=article.source,
                    confidence=article.classification_confidence,
                    metadata={
                        'url': article.url,
                        'published_date': article.published_date.isoformat(),
                        'keywords_matched': article.keywords_matched,
                        'celebrities_mentioned': article.metadata.get('celebrities_mentioned', [])
                    }
                )
                training_samples.append(sample)
        
        logger.info(f"Generated {len(training_samples)} training samples")
        return training_samples
    
    def get_classification_stats(self, articles: List[NewsArticle]) -> Dict[str, any]:
        """Get statistics about classification results."""
        stats = {
            'total_articles': len(articles),
            'classified_articles': sum(1 for a in articles if a.classified),
            'category_distribution': Counter(),
            'confidence_distribution': {
                'high (>0.7)': 0,
                'medium (0.4-0.7)': 0,
                'low (<0.4)': 0
            },
            'average_confidence': 0.0
        }
        
        confidences = []
        for article in articles:
            if article.classified and article.predicted_category:
                stats['category_distribution'][article.predicted_category.value] += 1
                
                if article.classification_confidence:
                    conf = article.classification_confidence
                    confidences.append(conf)
                    
                    if conf > 0.7:
                        stats['confidence_distribution']['high (>0.7)'] += 1
                    elif conf > 0.4:
                        stats['confidence_distribution']['medium (0.4-0.7)'] += 1
                    else:
                        stats['confidence_distribution']['low (<0.4)'] += 1
        
        if confidences:
            stats['average_confidence'] = sum(confidences) / len(confidences)
        
        return stats


# Enhanced LLM-based classifier (optional, requires LLM integration)
class LLMCelebrityEventClassifier(CelebrityEventClassifier):
    """LLM-enhanced classifier for more accurate categorization."""
    
    def __init__(self, config: DataCollectionConfig = None):
        super().__init__(config)
        self.use_llm = config.classification.use_llm_classification if config else False
        self.fallback_to_keyword = config.classification.fallback_to_keyword_matching if config else True
    
    async def _llm_classify(self, text: str) -> ClassificationResult:
        """Classify using LLM (placeholder for actual implementation)."""
        # This would integrate with your LLM system
        # For now, return keyword-based classification
        return await super().classify_article(NewsArticle(
            title=text, 
            url="", 
            source="", 
            published_date=datetime.now()
        ))
    
    async def classify_article(self, article: NewsArticle) -> ClassificationResult:
        """Classify using LLM with keyword fallback."""
        text_to_classify = article.title
        if article.summary:
            text_to_classify += " " + article.summary
        
        if self.use_llm:
            try:
                # Attempt LLM classification
                result = await self._llm_classify(text_to_classify)
                result.method = "llm"
                return result
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")
                if self.fallback_to_keyword:
                    logger.info("Falling back to keyword classification")
                    return await super().classify_article(article)
                else:
                    raise
        else:
            return await super().classify_article(article)


# Test function
async def test_classifier():
    """Test the celebrity event classifier."""
    from .config import DEFAULT_CONFIG
    
    classifier = CelebrityEventClassifier(DEFAULT_CONFIG)
    
    # Test headlines
    test_headlines = [
        "Taylor Swift and Travis Kelce Break Up After Year-Long Romance",
        "Brad Pitt Files for Divorce from Angelina Jolie",
        "Jennifer Lopez and Ben Affleck Engaged Again",
        "Kim Kardashian and Pete Davidson Spotted Fighting at Restaurant",
        "Britney Spears Announces Pregnancy with Third Child",
        "Johnny Depp Wins Lawsuit Against Amber Heard",
        "Kanye West Starts New Business Venture",
        "Selena Gomez Reveals Mental Health Struggles in Interview",
    ]
    
    print("Testing Celebrity Event Classifier...")
    
    for headline in test_headlines:
        article = NewsArticle(
            title=headline,
            url="test",
            source="Test",
            published_date=datetime.now()
        )
        
        result = await classifier.classify_article(article)
        print(f"\nHeadline: {headline}")
        print(f"Category: {result.category.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Keywords: {result.keywords_matched}")


if __name__ == "__main__":
    asyncio.run(test_classifier())