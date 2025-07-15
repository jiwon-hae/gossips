import asyncio
from collections import Counter

try:
    from .google_news import CelebrityGoogleNewsCollector
    from .utility.text_processing import extract_people
    from .search_config import Period
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from google_news import CelebrityGoogleNewsCollector
    from utility.text_processing import extract_people
    from search_config import Period

class TrendCollector:
    def __init__(self):
        self.gnews = CelebrityGoogleNewsCollector()
    
    async def get_trending_celeb(self, limit: int = 10):
        celeb_news = await self.gnews.search_celebrities_news(period=Period.YEARLY)
        
        # Extract people from all news articles
        all_people = []
        for news in celeb_news:
            people = extract_people(news.title)
            all_people.extend(people)
        
        # Count occurrences and return top k trending celebrities
        trending_people = [name for name, _ in Counter(all_people).most_common(limit)]
        return trending_people

async def main():
    trend = TrendCollector()
    trending_celebs = await trend.get_trending_celeb()
    print("Trending celebrities:", trending_celebs)

if __name__ == "__main__":
    asyncio.run(main())
    
