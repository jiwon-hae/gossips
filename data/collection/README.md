# Celebrity News Data Collection Pipeline

A comprehensive system for collecting celebrity news from Google News and classifying them into interpersonal events for training machine learning models.

## Features

- **Automated News Collection**: Uses the `gnews` library to collect celebrity news from Google News
- **Event Classification**: Classifies news into specific celebrity event categories (divorce, fights, lawsuits, etc.)
- **Celebrity Detection**: Automatically identifies celebrities mentioned in headlines
- **Multiple Export Formats**: Supports JSON, CSV, and Parquet export formats
- **Training Data Generation**: Automatically generates labeled training data for ML models
- **Configurable Pipeline**: Highly configurable for different collection strategies

## Celebrity Event Categories

The system classifies celebrity news into the following event categories:

### Relationship Events
- **Divorce**: Celebrity divorces and separations
- **Breakup**: Dating relationship breakups
- **Engagement**: Celebrity engagements
- **Marriage**: Celebrity weddings and marriages
- **Dating**: New celebrity relationships
- **Cheating**: Infidelity scandals
- **Reconciliation**: Getting back together

### Conflict Events
- **Feud**: Ongoing rivalries between celebrities
- **Fight**: Arguments and confrontations
- **Lawsuit**: Legal actions and court cases
- **Controversy**: Public controversies and criticism
- **Scandal**: Major scandals and exposés
- **Beef**: Hip-hop/music industry conflicts
- **Diss**: Public insults and call-outs

### Personal Events
- **Pregnancy**: Celebrity pregnancies
- **Birth**: Celebrity births
- **Death**: Celebrity deaths
- **Health Issue**: Health problems and scares
- **Addiction**: Substance abuse issues
- **Rehab**: Treatment and rehabilitation
- **Mental Health**: Mental health struggles

### Career Events
- **New Project**: New movies, albums, shows
- **Collaboration**: Celebrity collaborations
- **Award**: Awards and recognition
- **Nomination**: Award nominations
- **Retirement**: Career endings
- **Comeback**: Career comebacks

### Social Events
- **Party**: Celebrity parties and events
- **Red Carpet**: Fashion and red carpet moments
- **Vacation**: Celebrity vacations
- **Friendship**: Celebrity friendships
- **Family Drama**: Family conflicts

### Other Categories
- **Business Venture**: Celebrity business launches
- **Financial Trouble**: Financial problems
- **Charity**: Charitable activities
- **Social Media Drama**: Online controversies
- **Viral Moment**: Internet viral moments
- **Fashion Moment**: Fashion and style events

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install additional optional dependencies for enhanced features:
```bash
pip install spacy nltk
python -m spacy download en_core_web_sm
```

## Quick Start

### Basic Usage

```python
import asyncio
from data_collection.pipeline import CelebrityNewsDataPipeline

async def main():
    # Create pipeline with default settings
    pipeline = CelebrityNewsDataPipeline()
    
    # Collect and classify celebrity news
    articles, stats = await pipeline.run_collection()
    
    print(f"Collected {len(articles)} articles")
    print(f"Classified {stats.articles_classified} articles")

asyncio.run(main())
```

### Custom Configuration

```python
from data_collection.config import DataCollectionConfig
from data_collection.pipeline import CelebrityNewsDataPipeline

# Create custom configuration
config = DataCollectionConfig(
    max_articles_per_run=100,
    output_directory="my_celebrity_data",
    export_formats=["json", "csv"],
    collection_interval_hours=6
)

pipeline = CelebrityNewsDataPipeline(config)
articles, stats = await pipeline.run_collection()
```

### Targeted Celebrity Search

```python
from data_collection.gnews_scraper import CelebrityGNewsScraper

scraper = CelebrityGNewsScraper()

# Search for specific celebrity
articles = await scraper.search_celebrity_news(
    celebrity_name="Taylor Swift",
    max_results=10
)

# Search for specific event type
articles = await scraper.search_celebrity_news(
    event_type=EventCategory.DIVORCE,
    max_results=10
)
```

### Event Classification

```python
from data_collection.event_classifier import CelebrityEventClassifier
from data_collection.models import NewsArticle

classifier = CelebrityEventClassifier()

# Classify a single article
article = NewsArticle(
    title="Taylor Swift and Travis Kelce Break Up After Year-Long Romance",
    url="https://example.com",
    source="Entertainment News",
    published_date=datetime.now()
)

result = await classifier.classify_article(article)
print(f"Category: {result.category.value}")
print(f"Confidence: {result.confidence}")
```

## Command Line Usage

Run the pipeline from the command line:

```bash
# Basic collection
python -m data_collection.pipeline

# Custom parameters
python -m data_collection.pipeline --max-articles 500 --output-dir my_data --formats json csv

# Scheduled collection (runs continuously)
python -m data_collection.pipeline --scheduled
```

## Output Files

The pipeline generates several output files:

### Data Files
- `celebrity_news_YYYYMMDD_HHMMSS.json` - Complete article data
- `celebrity_news_YYYYMMDD_HHMMSS.csv` - Article data in CSV format
- `celebrity_news_YYYYMMDD_HHMMSS.parquet` - Article data in Parquet format

### Training Data
- `training_data_YYYYMMDD_HHMMSS.json` - Labeled training data
- `training_data_YYYYMMDD_HHMMSS.csv` - Training data in CSV format

### Statistics and Reports
- `collection_stats_YYYYMMDD_HHMMSS.json` - Collection statistics
- `classification_stats_YYYYMMDD_HHMMSS.json` - Classification statistics
- `report_YYYYMMDD_HHMMSS.txt` - Human-readable summary report

## Data Schema

### NewsArticle Fields
- `id`: Unique article identifier
- `title`: Article headline
- `url`: Article URL
- `source`: News source
- `published_date`: Publication date
- `summary`: Article summary (if available)
- `predicted_category`: Classified event category
- `classification_confidence`: Classification confidence score
- `keywords_matched`: Keywords that triggered classification
- `celebrities_mentioned`: Celebrities detected in the headline
- `collected_at`: Collection timestamp

### Training Data Fields
- `text`: Article headline/text
- `label`: Event category label
- `source`: News source
- `confidence`: Classification confidence
- `metadata`: Additional metadata

## Configuration Options

### DataCollectionConfig
```python
@dataclass
class DataCollectionConfig:
    collection_interval_hours: int = 6       # How often to collect
    max_articles_per_run: int = 500         # Max articles per collection
    days_to_collect: int = 30               # How many days back to search
    output_directory: str = "data/collected_news"
    export_formats: List[str] = ["json", "csv"]
    clean_html: bool = True                 # Clean HTML from content
    remove_duplicates: bool = True          # Remove duplicate articles
    min_headline_length: int = 10           # Minimum headline length
    max_headline_length: int = 200          # Maximum headline length
```

### EventClassificationConfig
```python
@dataclass
class EventClassificationConfig:
    categories: List[EventCategory]         # Categories to classify
    confidence_threshold: float = 0.7       # Minimum confidence threshold
    use_llm_classification: bool = True     # Use LLM for classification
    fallback_to_keyword_matching: bool = True
```

## Celebrity List

The system includes a curated list of popular celebrities for targeted searching:

- **Actors/Actresses**: Brad Pitt, Angelina Jolie, Jennifer Aniston, Leonardo DiCaprio, etc.
- **Musicians**: Taylor Swift, Ariana Grande, Justin Bieber, Beyoncé, etc.
- **Reality TV**: Kim Kardashian, Kylie Jenner, Paris Hilton, etc.
- **Newer Celebrities**: Olivia Rodrigo, Billie Eilish, Tom Holland, etc.

## Advanced Features

### Custom Celebrity Lists
```python
scraper = CelebrityGNewsScraper()
custom_celebrities = ["Celebrity Name 1", "Celebrity Name 2"]
articles = await scraper.collect_celebrity_specific_news(
    celebrity_names=custom_celebrities,
    max_per_celebrity=5
)
```

### Batch Processing
```python
# Process multiple event types
event_types = [EventCategory.DIVORCE, EventCategory.LAWSUIT, EventCategory.SCANDAL]
articles = await scraper.collect_event_specific_news(
    event_types=event_types,
    max_per_event=10
)
```

### Training Data Export
```python
# Generate high-quality training data
training_samples = classifier.generate_training_data(
    articles,
    min_confidence=0.7  # Only high-confidence classifications
)
```

## Rate Limiting and Ethics

The system implements respectful rate limiting:
- 1-2 second delays between requests
- Conservative limits on simultaneous requests
- Respects robots.txt and terms of service

## Machine Learning Integration

The generated training data can be used with popular ML frameworks:

### With scikit-learn
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load training data
df = pd.read_csv("training_data_YYYYMMDD_HHMMSS.csv")

# Prepare features and labels
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train classifier
classifier = LogisticRegression()
classifier.fit(X, y)
```

### With transformers
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use the labeled data to fine-tune a BERT model
# for celebrity event classification
```

## Troubleshooting

### Common Issues

1. **Rate Limiting**: If you get rate limited, increase the delay in `CelebrityGNewsScraper`
2. **No Articles Found**: Check your search queries and try broader terms
3. **Low Classification Confidence**: Review and expand the keyword lists for categories
4. **Memory Issues**: Reduce `max_articles_per_run` for large collections

### Logging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is for educational and research purposes. Please respect the terms of service of news sources and implement appropriate rate limiting.

## Future Enhancements

- [ ] Integration with additional news sources
- [ ] Advanced NLP for better celebrity name recognition
- [ ] Real-time classification API
- [ ] Web interface for data exploration
- [ ] Integration with social media platforms
- [ ] Sentiment analysis for events
- [ ] Relationship graph construction between celebrities