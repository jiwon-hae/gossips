"""
Main data collection pipeline for celebrity news and event classification.
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
    
import asyncio
import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from data.collection.gnews_scraper import CelebrityNewsAggregator
from data.collection.event_classifier import CelebrityEventClassifier
from data.collection.models import NewsArticle, CollectionStats, TrainingDataSample
from data.collection.config import DataCollectionConfig, DEFAULT_CONFIG, EventCategory

logger = logging.getLogger(__name__)

class CelebrityNewsDataPipeline:
    """Complete pipeline for collecting and classifying celebrity news."""
    
    def __init__(self, config: DataCollectionConfig = None):
        """Initialize the pipeline."""
        self.config = config or DEFAULT_CONFIG
        self.aggregator = CelebrityNewsAggregator()
        self.classifier = CelebrityEventClassifier(self.config)
        
        # Ensure output directory exists
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "pipeline.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    async def run_collection(self) -> tuple[List[NewsArticle], CollectionStats]:
        """Run the complete collection and classification pipeline."""
        logger.info("Starting celebrity news data collection pipeline")
        
        try:
            # Step 1: Collect news articles
            logger.info("Step 1: Collecting celebrity news articles")
            articles, collection_stats = await self.aggregator.collect_comprehensive_celebrity_news(
                max_articles=self.config.max_articles_per_run
            )
            
            if not articles:
                logger.warning("No articles collected")
                return [], collection_stats
            
            logger.info(f"Collected {len(articles)} articles")
            
            # Step 2: Classify articles
            logger.info("Step 2: Classifying articles by event type")
            classified_articles = await self.classifier.classify_articles(articles)
            
            # Update stats with classification info
            collection_stats.articles_classified = sum(1 for a in classified_articles if a.classified)
            
            # Calculate category distribution
            for article in classified_articles:
                if article.classified and article.predicted_category:
                    if article.predicted_category not in collection_stats.categories_distribution:
                        collection_stats.categories_distribution[article.predicted_category] = 0
                    collection_stats.categories_distribution[article.predicted_category] += 1
            
            logger.info(f"Classified {collection_stats.articles_classified} articles")
            
            # Step 3: Export data
            logger.info("Step 3: Exporting collected data")
            await self.export_data(classified_articles, collection_stats)
            
            # Step 4: Generate training data
            logger.info("Step 4: Generating training data")
            training_data = self.classifier.generate_training_data(
                classified_articles, 
                min_confidence=0.5
            )
            await self.export_training_data(training_data)
            
            logger.info("Pipeline completed successfully")
            return classified_articles, collection_stats
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    async def export_data(self, articles: List[NewsArticle], stats: CollectionStats):
        """Export collected data in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export articles
        for format_type in self.config.export_formats:
            try:
                if format_type == "json":
                    await self._export_json(articles, f"celebrity_news_{timestamp}.json")
                elif format_type == "csv":
                    await self._export_csv(articles, f"celebrity_news_{timestamp}.csv")
                elif format_type == "parquet":
                    await self._export_parquet(articles, f"celebrity_news_{timestamp}.parquet")
                
                logger.info(f"Exported data in {format_type} format")
                
            except Exception as e:
                logger.error(f"Failed to export in {format_type} format: {str(e)}")
        
        # Export stats
        stats_file = self.output_dir / f"collection_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
        
        # Export classification stats
        classification_stats = self.classifier.get_classification_stats(articles)
        classification_file = self.output_dir / f"classification_stats_{timestamp}.json"
        with open(classification_file, 'w') as f:
            json.dump(classification_stats, f, indent=2)
    
    async def _export_json(self, articles: List[NewsArticle], filename: str):
        """Export articles as JSON."""
        filepath = self.output_dir / filename
        
        data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_articles": len(articles),
                "pipeline_config": {
                    "max_articles_per_run": self.config.max_articles_per_run,
                    "classification_threshold": self.config.classification.confidence_threshold
                }
            },
            "articles": [article.to_dict() for article in articles]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def _export_csv(self, articles: List[NewsArticle], filename: str):
        """Export articles as CSV."""
        filepath = self.output_dir / filename
        
        fieldnames = [
            'id', 'title', 'url', 'source', 'published_date', 'summary',
            'predicted_category', 'classification_confidence', 'classification_method',
            'keywords_matched', 'celebrities_mentioned', 'collected_at'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for article in articles:
                row = {
                    'id': article.id,
                    'title': article.title,
                    'url': article.url,
                    'source': article.source,
                    'published_date': article.published_date.isoformat(),
                    'summary': article.summary or '',
                    'predicted_category': article.predicted_category.value if article.predicted_category else '',
                    'classification_confidence': article.classification_confidence or 0,
                    'classification_method': article.classification_method or '',
                    'keywords_matched': ', '.join(article.keywords_matched) if article.keywords_matched else '',
                    'celebrities_mentioned': ', '.join(article.metadata.get('celebrities_mentioned', [])),
                    'collected_at': article.collected_at.isoformat()
                }
                writer.writerow(row)
    
    async def _export_parquet(self, articles: List[NewsArticle], filename: str):
        """Export articles as Parquet."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas or pyarrow not installed, skipping parquet export")
            return
        
        filepath = self.output_dir / filename
        
        # Convert to DataFrame
        data = []
        for article in articles:
            data.append({
                'id': article.id,
                'title': article.title,
                'url': article.url,
                'source': article.source,
                'published_date': article.published_date,
                'summary': article.summary or '',
                'predicted_category': article.predicted_category.value if article.predicted_category else '',
                'classification_confidence': article.classification_confidence or 0,
                'classification_method': article.classification_method or '',
                'keywords_matched': article.keywords_matched,
                'celebrities_mentioned': article.metadata.get('celebrities_mentioned', []),
                'collected_at': article.collected_at
            })
        
        df = pd.DataFrame(data)
        df.to_parquet(filepath, index=False)
    
    async def export_training_data(self, training_samples: List[TrainingDataSample]):
        """Export training data for ML model training."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export as JSON
        training_file = self.output_dir / f"training_data_{timestamp}.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_samples": len(training_samples),
                    "categories": list(set(sample.label.value for sample in training_samples))
                },
                "samples": [sample.to_dict() for sample in training_samples]
            }
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Export as CSV for easy loading in ML frameworks
        csv_file = self.output_dir / f"training_data_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'label', 'source', 'confidence'])
            writer.writeheader()
            
            for sample in training_samples:
                writer.writerow({
                    'text': sample.text,
                    'label': sample.label.value,
                    'source': sample.source,
                    'confidence': sample.confidence
                })
        
        logger.info(f"Exported {len(training_samples)} training samples")
    
    async def run_scheduled_collection(self):
        """Run collection on schedule (for production use)."""
        while True:
            try:
                await self.run_collection()
                
                # Wait for next collection interval
                wait_hours = self.config.collection_interval_hours
                logger.info(f"Waiting {wait_hours} hours until next collection...")
                await asyncio.sleep(wait_hours * 3600)
                
            except Exception as e:
                logger.error(f"Scheduled collection failed: {str(e)}")
                # Wait 1 hour before retry
                await asyncio.sleep(3600)
    
    def generate_report(self, articles: List[NewsArticle], stats: CollectionStats) -> str:
        """Generate a summary report of the collection."""
        classification_stats = self.classifier.get_classification_stats(articles)
        
        report = f"""
Celebrity News Data Collection Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== COLLECTION STATISTICS ===
Total Articles Found: {stats.total_articles_found}
Articles Collected: {stats.articles_collected}
Articles Classified: {stats.articles_classified}
Articles Skipped: {stats.articles_skipped}
Collection Duration: {stats.duration_seconds:.1f} seconds
Articles per Second: {stats.articles_per_second:.2f}

=== CLASSIFICATION RESULTS ===
Classification Success Rate: {(stats.articles_classified / stats.articles_collected * 100):.1f}%
Average Confidence: {classification_stats['average_confidence']:.2f}

Category Distribution:
"""
        
        for category, count in classification_stats['category_distribution'].most_common():
            percentage = (count / stats.articles_classified * 100) if stats.articles_classified > 0 else 0
            report += f"  {category}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
Confidence Distribution:
  High Confidence (>0.7): {classification_stats['confidence_distribution']['high (>0.7)']}
  Medium Confidence (0.4-0.7): {classification_stats['confidence_distribution']['medium (0.4-0.7)']}
  Low Confidence (<0.4): {classification_stats['confidence_distribution']['low (<0.4)']}

=== TOP CATEGORIES ===
"""
        
        top_categories = list(classification_stats['category_distribution'].most_common(5))
        for category, count in top_categories:
            report += f"  {category}: {count} articles\n"
        
        return report


# CLI interface
async def main():
    """Main function for running the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Celebrity News Data Collection Pipeline")
    parser.add_argument("--max-articles", type=int, default=200, help="Maximum articles to collect")
    parser.add_argument("--output-dir", type=str, default="data/celebrity_news", help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["json", "csv"], help="Export formats")
    parser.add_argument("--scheduled", action="store_true", help="Run on schedule")
    
    args = parser.parse_args()
    
    # Create config
    config = DataCollectionConfig(
        max_articles_per_run=args.max_articles,
        output_directory=args.output_dir,
        export_formats=args.formats
    )
    
    # Create and run pipeline
    pipeline = CelebrityNewsDataPipeline(config)
    
    if args.scheduled:
        logger.info("Starting scheduled collection...")
        await pipeline.run_scheduled_collection()
    else:
        logger.info("Running one-time collection...")
        articles, stats = await pipeline.run_collection()
        
        # Generate and save report
        report = pipeline.generate_report(articles, stats)
        report_file = Path(args.output_dir) / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        logger.info(f"Report saved to {report_file}")


if __name__ == "__main__":
    asyncio.run(main())