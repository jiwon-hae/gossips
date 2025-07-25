
import json
import logging
import time
import asyncio
from uuid import uuid4

from pathlib import Path
from googlenewsdecoder import gnewsdecoder
from datetime import date
from typing import List, Optional
from gnews import GNews
from dataclasses import dataclass
from newspaper import Article

try:
    from .search_config import Period
    from ...ingestion.enums import *

except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    ))

    from data.collectors.search_config import Period
    from ingestion.enums import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class GoogleNews:
    title: str
    description: str
    published_date: date
    url: str
    publisher: str


class GoogleNewsCollector:
    def __init__(self, country: str = "US", language: str = "en"):
        self.gnews = GNews(language=language, country=country)
        self.rate_limit_delay = 1.0
        self.last_request_time = 0.0
        
        DATA_DIR = Path(__file__).resolve().parent.parent.parent
        self.base_path = DATA_DIR / 'documents' / "articles"
        self.base_path.mkdir(parents=True, exist_ok=True)

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

    def resolve_news_url(self, gn_url: str) -> str:
        """
        Extract and base64-decode the real article URL embedded in the
        Google News RSS link. No HTTP request needed.
        """
        decoded_url = gnewsdecoder(gn_url)
        return decoded_url['decoded_url']

    def convert(self, news_list: List[dict]) -> List[GoogleNews]:
        """Convert Google News API output format to GoogleNews objects."""
        converted_news = []

        for news_item in news_list:
            try:
                google_news = GoogleNews(
                    title=news_item.get('title', ''),
                    description=news_item.get('description', ''),
                    published_date=news_item.get('published date', ''),
                    url=self.resolve_news_url(news_item.get('url', '')),
                    publisher=news_item.get('publisher', {}).get('title', ''),
                )
                converted_news.append(google_news)
            except Exception as e:
                logger.error(f"Error converting news item: {e}")
                continue

        return converted_news

    async def get_news(
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
        self.configure(period=period, max_results=max_results,
                       start_date=start_date, end_date=end_date)
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
        self.configure(period=period, max_results=max_results,
                       start_date=start_date, end_date=end_date)
        return await self.convert(self.gnews.get_top_news())

    async def get_full_article(self, url) -> Article:
        return self.gnews.get_full_article(url)
    
    async def save_article(self, url : str, metadata : dict):
        doc_id = uuid4().hex
        article = Article(url)
        try:
            await self._rate_limit()
            article.build()
            data = {
                'title': article.title,
                "id" : doc_id,
                "content" : article.text,
                "keywords": article.keywords,
                'url': url,
                
                **metadata,   
            }
            
            celeb_path = self.base_path / metadata['celeb']
            celeb_path.mkdir(parents=True, exist_ok=True)
            
            with open(celeb_path / f'{article.title}.json', "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save articles JSON {metadata['celeb']}({e})")
            


class CelebrityGoogleNewsCollector(GoogleNewsCollector):
    def __init__(self, country: str = "US", language: str = "en"):
        super().__init__(country=country, language=language)

    async def search_celebrities_news(
        self,
        max_results: Optional[int] = None,
        period: Period = Period.WEEKLY,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
        self.configure(period=period, max_results=max_results,
                       start_date=start_date, end_date=end_date)
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
                celebrity_news = await self.get_news(celebrity, max_results=max_results, period=period, start_date=start_date, end_date=end_date)
                if celebrity_news:
                    results.extend(celebrity_news)
            except Exception as e:
                logger.error(f"Error searching for {celebrity}: {e}")

        return results

    async def search_events(
        self,
        events: List[Event] = Event.__members__,
        max_results: int = 10,
        period: Period = Period.WEEKLY,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
        results = []

        try:
            query = " OR ".join(events)
            event_news = await self.get_news(query, max_results=max_results, period=period, start_date=start_date, end_date=end_date)
            if event_news:
                results.extend(event_news)
        except Exception as e:
            logger.error(f"Error searching for {events}: {e}")

        return results


async def main():
    gnews_collector = CelebrityGoogleNewsCollector()
    news = await gnews_collector.get_news("Justin Bieber", max_results=10)


if __name__ == '__main__':
    asyncio.run(main())
