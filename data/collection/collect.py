import asyncio
from pipeline import CelebrityNewsDataPipeline

async def collect_gossips():
    pipeline = CelebrityNewsDataPipeline()

    # Run collection
    articles, stats = await pipeline.run_collection()

    print(f"Collected {len(articles)} articles")
    print(f"Collection took {stats.duration_seconds:.1f} seconds")

    # Show some examples
    print("\nSample Headlines:")
    for i, article in enumerate(articles[:5]):
        category = article.predicted_category.value if article.predicted_category else "unknown"
        confidence = article.classification_confidence or 0
        print(f"{i+1}. [{category}] ({confidence:.2f}) {article.title}")

    return articles, stats

if __name__ == "__main__":
    asyncio.run(collect_gossips())
    
