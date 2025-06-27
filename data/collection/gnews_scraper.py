"""
Celebrity news scraper using gnews library for Google News access.
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    

import asyncio
import logging
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import re

try:
    from gnews import GNews
except ImportError:
    print("Please install gnews: pip install gnews")
    raise

from data.collection.models import NewsArticle, CollectionStats
from data.collection.config import EventCategory

logger = logging.getLogger(__name__)


@dataclass
class CelebritySearchQuery:
    """Search query configuration for celebrity news."""
    celebrity_name: Optional[str] = None
    event_type: Optional[EventCategory] = None
    keywords: List[str] = None
    max_results: int = 10
    period: str = "7d"  # 1h, 24h, 7d, 1m, 1y
    
    def generate_query(self) -> str:
        """Generate search query string."""
        parts = []
        
        if self.celebrity_name:
            parts.append(f'"{self.celebrity_name}"')
        
        if self.event_type and self.keywords:
            # Add event-specific keywords
            event_keywords = " OR ".join(self.keywords[:3])  # Limit to avoid too complex queries
            parts.append(f"({event_keywords})")
        
        # Always include celebrity-related terms
        parts.append("(celebrity OR star OR actor OR actress OR singer OR musician)")
        
        return " AND ".join(parts)


class CelebrityGNewsScraper:
    """Scraper for celebrity news using gnews library."""
    
    def __init__(self, language: str = "en", country: str = "US"):
        """Initialize the scraper."""
        self.gnews = GNews(language=language, country=country)
        self.rate_limit_delay = 1.0  # Seconds between requests
        self.last_request_time = 0.0
        
        # List of popular celebrities for targeted searching
        self.celebrity_list = [
            # Actors/Actresses
            "Brad Pitt", "Angelina Jolie", "Jennifer Aniston", "Leonardo DiCaprio",
            "Scarlett Johansson", "Ryan Reynolds", "Blake Lively", "Tom Cruise",
            "Jennifer Lawrence", "Emma Stone", "Ryan Gosling", "Margot Robbie",
            "Chris Evans", "Robert Downey Jr", "Zendaya", "Timothée Chalamet",
            
            # Musicians
            "Taylor Swift", "Ariana Grande", "Justin Bieber", "Selena Gomez",
            "Beyonce", "Jay-Z", "Kanye West", "Kim Kardashian", "Drake",
            "Rihanna", "Lady Gaga", "Bruno Mars", "Ed Sheeran", "Adele",
            
            # Reality TV / Influencers
            "Kylie Jenner", "Kendall Jenner", "Khloe Kardashian", "Kourtney Kardashian",
            "Paris Hilton", "Nicole Richie", "Britney Spears", "Justin Timberlake",
            
            # Newer celebrities
            "Olivia Rodrigo", "Billie Eilish", "Dua Lipa", "Harry Styles",
            "Tom Holland", "Anya Taylor-Joy", "Florence Pugh", "Sydney Sweeney"
        ]
        
        # Event-specific search terms
        self.event_search_terms = {
            EventCategory.DIVORCE: ["divorce", "split", "separation", "custody"],
            EventCategory.BREAKUP: ["breakup", "broke up", "ended relationship", "split"],
            EventCategory.ENGAGEMENT: ["engaged", "engagement", "proposal", "ring"],
            EventCategory.MARRIAGE: ["married", "wedding", "ceremony", "tied the knot"],
            EventCategory.DATING: ["dating", "relationship", "boyfriend", "girlfriend"],
            EventCategory.FEUD: ["feud", "rivalry", "beef", "tension"],
            EventCategory.FIGHT: ["fight", "argument", "confrontation", "spat"],
            EventCategory.LAWSUIT: ["lawsuit", "sued", "legal action", "court"],
            EventCategory.SCANDAL: ["scandal", "controversy", "exposed", "leaked"],
            EventCategory.PREGNANCY: ["pregnant", "pregnancy", "expecting", "baby"],
            EventCategory.BIRTH: ["birth", "born", "baby", "welcomed"],
            EventCategory.DEATH: ["died", "death", "passed away", "funeral"],
        }
    
    async def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize article title."""
        # Remove common prefixes/suffixes
        title = re.sub(r'^(EXCLUSIVE|BREAKING|UPDATE|PHOTOS?):?\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*-\s*(TMZ|People|Entertainment Tonight|Us Weekly|E!).*$', '', title, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        title = re.sub(r'[!]{2,}', '!', title)
        title = re.sub(r'[?]{2,}', '?', title)
        
        return title.strip()
    
    def _extract_celebrities_from_title(self, title: str) -> List[str]:
        """Extract celebrity names mentioned in the title."""
        found_celebrities = []
        title_lower = title.lower()
        
        for celebrity in self.celebrity_list:
            # Check for full name
            if celebrity.lower() in title_lower:
                found_celebrities.append(celebrity)
            else:
                # Check for last name only (for common celebrities)
                last_name = celebrity.split()[-1].lower()
                if len(last_name) > 3 and last_name in title_lower:
                    found_celebrities.append(celebrity)
        
        return list(set(found_celebrities))  # Remove duplicates
    
    async def search_celebrity_news(self, 
                                  celebrity_name: Optional[str] = None,
                                  event_type: Optional[EventCategory] = None,
                                  max_results: int = 10,
                                  period: str = "7d") -> List[NewsArticle]:
        """Search for celebrity news with optional filters."""
        await self._rate_limit()
        
        articles = []
        
        try:
            # Configure search parameters
            self.gnews.period = period
            self.gnews.max_results = max_results
            
            # Generate search query
            query_config = CelebritySearchQuery(
                celebrity_name=celebrity_name,
                event_type=event_type,
                keywords=self.event_search_terms.get(event_type, []) if event_type else [],
                max_results=max_results,
                period=period
            )
            
            search_query = query_config.generate_query()
            logger.info(f"Searching with query: {search_query}")
            
            # Perform search
            search_results = self.gnews.get_news(search_query)
            
            for result in search_results:
                try:
                    article = self._convert_to_news_article(result, celebrity_name, event_type)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.error(f"Error converting article: {str(e)}")
                    continue
            
            logger.info(f"Found {len(articles)} articles for query: {search_query}")
            
        except Exception as e:
            logger.error(f"Error searching celebrity news: {str(e)}")
        
        return articles
    
    def _convert_to_news_article(self, 
                                gnews_result: Dict[str, Any],
                                celebrity_name: Optional[str] = None,
                                event_type: Optional[EventCategory] = None) -> Optional[NewsArticle]:
        """Convert gnews result to NewsArticle."""
        try:
            title = gnews_result.get('title', '').strip()
            if not title:
                return None
            
            # Clean the title
            title = self._clean_title(title)
            
            url = gnews_result.get('url', '').strip()
            if not url:
                return None
            
            # Parse publication date
            published_date = datetime.now()
            if 'published date' in gnews_result:
                try:
                    pub_date_str = gnews_result['published date']
                    # gnews returns dates in format like "Tue, 19 Dec 2023 10:30:00 GMT"
                    from dateutil import parser
                    published_date = parser.parse(pub_date_str)
                except Exception as e:
                    logger.debug(f"Could not parse date '{gnews_result.get('published date')}': {e}")
            
            # Extract publisher/source
            publisher = gnews_result.get('publisher', {})
            source = "Unknown"
            if isinstance(publisher, dict):
                source = publisher.get('title', 'Unknown')
            elif isinstance(publisher, str):
                source = publisher
            
            # Extract celebrities mentioned
            celebrities_mentioned = self._extract_celebrities_from_title(title)
            if celebrity_name and celebrity_name not in celebrities_mentioned:
                celebrities_mentioned.append(celebrity_name)
            
            # Create article
            article = NewsArticle(
                title=title,
                url=url,
                source=source,
                published_date=published_date,
                summary=None,  # gnews doesn't provide summary
                collection_source="gnews",
                category="celebrity"
            )
            
            # Add metadata
            article.metadata.update({
                'celebrities_mentioned': celebrities_mentioned,
                'search_celebrity': celebrity_name,
                'search_event_type': event_type.value if event_type else None,
                'gnews_raw': gnews_result
            })
            
            return article
            
        except Exception as e:
            logger.error(f"Error converting gnews result to article: {str(e)}")
            return None
    
    async def collect_trending_celebrity_news(self, max_results: int = 50) -> List[NewsArticle]:
        """Collect trending celebrity news."""
        await self._rate_limit()
        
        articles = []
        
        try:
            # Search for general celebrity news
            general_queries = [
                "celebrity news",
                "hollywood news",
                "entertainment news",
                "celebrity gossip",
                "celebrity drama"
            ]
            
            for query in general_queries:
                try:
                    self.gnews.max_results = max_results // len(general_queries)
                    results = self.gnews.get_news(query)
                    
                    for result in results:
                        article = self._convert_to_news_article(result)
                        if article:
                            articles.append(article)
                    
                    await self._rate_limit()
                    
                except Exception as e:
                    logger.error(f"Error with query '{query}': {str(e)}")
                    continue
            
            logger.info(f"Collected {len(articles)} trending celebrity articles")
            
        except Exception as e:
            logger.error(f"Error collecting trending news: {str(e)}")
        
        return articles
    
    async def collect_celebrity_specific_news(self, 
                                            celebrity_names: List[str] = None,
                                            max_per_celebrity: int = 5) -> List[NewsArticle]:
        """Collect news for specific celebrities."""
        if not celebrity_names:
            celebrity_names = self.celebrity_list[:20]  # Use top 20 celebrities
        
        all_articles = []
        
        for celebrity in celebrity_names:
            try:
                logger.info(f"Collecting news for {celebrity}")
                articles = await self.search_celebrity_news(
                    celebrity_name=celebrity,
                    max_results=max_per_celebrity
                )
                all_articles.extend(articles)
                
                await self._rate_limit()
                
            except Exception as e:
                logger.error(f"Error collecting news for {celebrity}: {str(e)}")
                continue
        
        logger.info(f"Collected {len(all_articles)} articles for {len(celebrity_names)} celebrities")
        return all_articles
    
    async def collect_event_specific_news(self, 
                                        event_types: List[EventCategory] = None,
                                        max_per_event: int = 10) -> List[NewsArticle]:
        """Collect news for specific event types."""
        if not event_types:
            event_types = [
                EventCategory.DIVORCE, EventCategory.BREAKUP, EventCategory.ENGAGEMENT,
                EventCategory.MARRIAGE, EventCategory.DATING, EventCategory.FEUD,
                EventCategory.FIGHT, EventCategory.LAWSUIT, EventCategory.SCANDAL
            ]
        
        all_articles = []
        
        for event_type in event_types:
            try:
                logger.info(f"Collecting news for event type: {event_type.value}")
                articles = await self.search_celebrity_news(
                    event_type=event_type,
                    max_results=max_per_event
                )
                all_articles.extend(articles)
                
                await self._rate_limit()
                
            except Exception as e:
                logger.error(f"Error collecting news for event {event_type.value}: {str(e)}")
                continue
        
        logger.info(f"Collected {len(all_articles)} articles for {len(event_types)} event types")
        return all_articles


class CelebrityNewsAggregator:
    """Aggregator for celebrity news from multiple strategies."""
    
    def __init__(self):
        self.scraper = CelebrityGNewsScraper()
    
    async def collect_comprehensive_celebrity_news(self, 
                                                 max_articles: int = 200) -> tuple[List[NewsArticle], CollectionStats]:
        """Collect celebrity news using multiple strategies."""
        stats = CollectionStats(start_time=datetime.now())
        all_articles = []
        
        try:
            # Strategy 1: Trending celebrity news
            logger.info("Collecting trending celebrity news...")
            trending_articles = await self.scraper.collect_trending_celebrity_news(max_results=50)
            all_articles.extend(trending_articles)
            stats.total_articles_found += len(trending_articles)
            
            # Strategy 2: Celebrity-specific news
            logger.info("Collecting celebrity-specific news...")
            celebrity_articles = await self.scraper.collect_celebrity_specific_news(max_per_celebrity=3)
            all_articles.extend(celebrity_articles)
            stats.total_articles_found += len(celebrity_articles)
            
            # Strategy 3: Event-specific news
            logger.info("Collecting event-specific news...")
            event_articles = await self.scraper.collect_event_specific_news(max_per_event=5)
            all_articles.extend(event_articles)
            stats.total_articles_found += len(event_articles)
            
            # Remove duplicates based on URL
            seen_urls = set()
            unique_articles = []
            for article in all_articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    unique_articles.append(article)
                else:
                    stats.articles_skipped += 1
            
            # Limit to max articles
            if len(unique_articles) > max_articles:
                unique_articles = unique_articles[:max_articles]
                stats.articles_skipped += len(all_articles) - max_articles
            
            stats.articles_collected = len(unique_articles)
            stats.end_time = datetime.now()
            stats.sources_processed = ["gnews"]
            
            logger.info(f"Collection completed: {stats.articles_collected} unique articles")
            return unique_articles, stats
            
        except Exception as e:
            logger.error(f"Error in celebrity news aggregation: {str(e)}")
            stats.errors_encountered += 1
            stats.end_time = datetime.now()
            return [], stats


# Test function
async def test_celebrity_scraper():
    """Test function for celebrity news scraper."""
    scraper = CelebrityGNewsScraper()
    
    print("Testing celebrity news scraper...")
    
    # Test 1: Search for specific celebrity
    print("\n1. Testing celebrity-specific search...")
    articles = await scraper.search_celebrity_news("Taylor Swift", max_results=5)
    for i, article in enumerate(articles[:3]):
        print(f"   {i+1}. {article.title}")
        print(f"      Source: {article.source}")
        print(f"      Celebrities: {article.metadata.get('celebrities_mentioned', [])}")
    
    # Test 2: Search for event type
    print("\n2. Testing event-specific search...")
    articles = await scraper.search_celebrity_news(event_type=EventCategory.DIVORCE, max_results=5)
    for i, article in enumerate(articles[:3]):
        print(f"   {i+1}. {article.title}")
        print(f"      Source: {article.source}")
    
    # Test 3: Trending news
    print("\n3. Testing trending celebrity news...")
    articles = await scraper.collect_trending_celebrity_news(max_results=5)
    for i, article in enumerate(articles[:3]):
        print(f"   {i+1}. {article.title}")
        print(f"      Source: {article.source}")


if __name__ == "__main__":
    # Test the scraper
    asyncio.run(test_celebrity_scraper())