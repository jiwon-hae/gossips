
import logging
import time
import asyncio


from datetime import date
from typing import List, Optional
from gnews import GNews
from dataclasses import dataclass
from newspaper import Article

try:
    from .search_config import Period, EventCategory
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    from search_config import Period, EventCategory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class GoogleNews:
    title : str
    description: str
    published_date : date
    url : str
    publisher : str
    


class GNewCollector:
    def __init__(self, country: str = "US", language: str = "en"):
        self.gnews = GNews(language=language, country=country)
        self.rate_limit_delay = 1.0
        self.last_request_time = 0.0

    async def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time = time.time()
    
    def configure(self, 
        max_results: Optional[int] = None,
        period: Optional[str] = Period.WEEKLY,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None):
        self.gnews.period = period
        
        if max_results is not None:
            self.gnews.max_results = max_results

        if start_date is not None:
            self.gnews.start_date = (
                start_date.year, start_date.month, start_date.day)

        if end_date is not None:
            self.gnews.end_date = (end_date.year, end_date.month, end_date.day)
    
    def convert(self, news_list: List[dict]) -> List[GoogleNews]:
        """Convert Google News API output format to GoogleNews objects."""
        converted_news = []
        
        for news_item in news_list:
            try:
                google_news = GoogleNews(
                    title=news_item.get('title', ''),
                    description=news_item.get('description', ''),
                    published_date=news_item.get('published date', ''),
                    url=news_item.get('url', ''),
                    publisher=news_item.get('publisher', {}).get('title', ''),
                )
                converted_news.append(google_news)
            except Exception as e:
                logger.error(f"Error converting news item: {e}")
                continue
        
        return converted_news
        

    async def search(
        self,
        search_query: str,
        max_results: Optional[int] = None,
        period: Optional[str] = Period.WEEKLY,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[GoogleNews]:
        # Assert that if date range is provided, period should be None
        if start_date is not None and end_date is not None:
            assert period is None, "Cannot use both period and date range"

        await self._rate_limit()
        self.configure(period=period, max_results=max_results, start_date=start_date, end_date=end_date)
        self.gnews.period = period
        return self.convert(self.gnews.get_news(search_query))

    async def search_top_news(
        self,
        max_results: int = 10,
        period: Period = Period.WEEKLY,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[GoogleNews]:
        
        await self._rate_limit()
        self.configure(period=period, max_results=max_results, start_date=start_date, end_date=end_date)
        return self.convert(self.gnews.get_top_news())
    
    async def get_full_article(self, url) -> Article:
        await self._rate_limit()
        
        return self.gnews.get_full_article(url)
        
        

class CelebrityGNewsCollector(GNewCollector):
    def __init__(self, country: str = "US", language: str = "en"):
        super().__init__(country=country, language=language)
        
    async def search_celebrities(
        self,
        max_results: Optional[int] = None,
        period: Period = Period.WEEKLY,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
        self.configure(period=period, max_results=max_results, start_date=start_date, end_date=end_date)
        return self.convert(self.gnews.get_news_by_topic('CELEBRITIES'))
        

    async def search_target_celebrities(
        self,
        celebrities: List[str],
        max_results: Optional[int] = None,
        period: Period = Period.WEEKLY,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
        if not celebrities:
            return []

        results = []

        for celebrity in celebrities:
            try:
                # Search using parent class method
                celebrity_news = await self.search(celebrity, max_results=max_results, period=period, start_date=start_date, end_date=end_date)
                if celebrity_news:
                    results.extend(celebrity_news)
            except Exception as e:
                logger.error(f"Error searching for {celebrity}: {e}")

        return results

    async def search_events(
        self,
        events: List[EventCategory] = EventCategory.__members__,
        max_results: int = 10,
        period: Period = Period.WEEKLY,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
        results = []
        
        try:
            query = " OR ".join(events)
            event_news = await self.search(query, max_results=max_results, period=period, start_date=start_date, end_date=end_date)
            if event_news:
                results.extend(event_news)
        except Exception as e:
            logger.error(f"Error searching for {events}: {e}")

        return results

    


async def main():
    gnews_collector = CelebrityGNewsCollector()
    
    result = await gnews_collector.search_celebrities(start_date=date(year=2025, month = 7, day=12))
    print(result[0])
    
    article = await gnews_collector.get_full_article(result[0].url)
    print(article.keywords)
    


if __name__ == '__main__':
    asyncio.run(main())
