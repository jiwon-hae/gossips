#!/usr/bin/env python3
"""
Example script showing how to use the trained celebrity news classifier.
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import asyncio
import logging
from datetime import datetime

from data.predictor.ml_classifier import CelebrityMLClassifier, TrainingConfig
from data.collection.models import NewsArticle
from data.collection.config import EventCategory

logger = logging.getLogger(__name__)


async def predict_example():
    """Example of using trained classifier for prediction."""
    print("Celebrity News Classifier - Prediction Example")
    print("=" * 50)
    
    # Try to load a trained model
    try:
        config = TrainingConfig(model_type="random_forest")
        classifier = CelebrityMLClassifier(config)
        classifier.load_model("models/celebrity_classifier_random_forest.pkl")
        print("✓ Loaded trained Random Forest model")
    except FileNotFoundError:
        print("⚠ No trained model found. Please run train_classifier.py first.")
        print("For demonstration, we'll create a simple example with synthetic data.")
        return
    
    # Example headlines to classify
    test_headlines = [
        "Taylor Swift and Travis Kelce Break Up After Year-Long Romance",
        "Kim Kardashian Files for Divorce from Pete Davidson After 9 Months",
        "Jennifer Lopez and Ben Affleck Announce Engagement",
        "Brad Pitt Wins Custody Battle Against Angelina Jolie",
        "Johnny Depp Files $50M Lawsuit Against Amber Heard",
        "Britney Spears Announces Pregnancy with Third Child",
        "Kanye West and Drake End Years-Long Feud",
        "Selena Gomez Reveals Mental Health Struggles in New Interview",
        "Leonardo DiCaprio Spotted with New Mystery Woman in Malibu",
        "Rihanna Launches New Luxury Fashion Brand",
        "Chris Hemsworth Takes Break from Acting Due to Health Concerns",
        "Ariana Grande and Dalton Gomez Secretly Marry in Italy"
    ]
    
    print(f"\\nClassifying {len(test_headlines)} celebrity news headlines...")
    print("-" * 70)
    
    # Classify each headline
    results = classifier.predict(test_headlines)
    
    for headline, result in zip(test_headlines, results):
        print(f"Headline: {headline}")
        print(f"Category: {result.category.value}")
        print(f"Confidence: {result.confidence:.3f}")
        
        # Show top 3 predictions if available
        if result.raw_scores:
            sorted_scores = sorted(result.raw_scores.items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
            print("Top predictions:")
            for category, score in sorted_scores:
                print(f"  {category.value}: {score:.3f}")
        
        print("-" * 70)
    
    # Demonstrate article-based prediction
    print("\\nDemonstrating article-based prediction...")
    print("-" * 50)
    
    sample_article = NewsArticle(
        title="Celebrity Couple Announces Shock Divorce After Secret Wedding",
        url="https://example.com/celebrity-divorce",
        source="Entertainment News",
        published_date=datetime.now(),
        summary="A famous celebrity couple shocked fans by announcing their divorce just months after their secret wedding ceremony."
    )
    
    result = classifier.predict_article(sample_article)
    print(f"Article Title: {sample_article.title}")
    print(f"Predicted Category: {result.category.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Method: {result.method}")
    
    # Show feature importance if available
    if classifier.feature_importance:
        print("\\nTop 10 Most Important Features in the Model:")
        print("-" * 50)
        top_features = classifier.get_top_features(10)
        for feature, importance in top_features:
            print(f"{feature:25}: {importance:.4f}")


async def batch_prediction_example():
    """Example of batch prediction on multiple articles."""
    print("\\n" + "=" * 50)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 50)
    
    # Load classifier
    try:
        config = TrainingConfig(model_type="random_forest")
        classifier = CelebrityMLClassifier(config)
        classifier.load_model("models/celebrity_classifier_random_forest.pkl")
    except FileNotFoundError:
        print("No trained model found for batch prediction example.")
        return
    
    # Create sample articles
    articles = [
        NewsArticle(
            title="Pop Star Announces Surprise Engagement to Actor",
            url="https://example.com/1",
            source="Celebrity News",
            published_date=datetime.now(),
            summary="A popular pop star surprised fans by announcing her engagement."
        ),
        NewsArticle(
            title="Hollywood Couple Files for Divorce After 10 Years",
            url="https://example.com/2", 
            source="Entertainment Weekly",
            published_date=datetime.now(),
            summary="After a decade of marriage, the couple has decided to separate."
        ),
        NewsArticle(
            title="Actor Welcomes First Child with Partner",
            url="https://example.com/3",
            source="People Magazine", 
            published_date=datetime.now(),
            summary="The couple welcomed their first child together this week."
        ),
        NewsArticle(
            title="Singer Involved in Legal Battle with Former Manager",
            url="https://example.com/4",
            source="Billboard",
            published_date=datetime.now(),
            summary="The legal dispute centers around contract violations."
        )
    ]
    
    print(f"Classifying {len(articles)} articles...")
    
    # Predict for all articles
    for article in articles:
        result = classifier.predict_article(article)
        
        print(f"\\nTitle: {article.title}")
        print(f"Source: {article.source}")
        print(f"Predicted Category: {result.category.value}")
        print(f"Confidence: {result.confidence:.3f}")


def check_model_performance():
    """Check the performance of trained models."""
    print("\\n" + "=" * 50)
    print("MODEL PERFORMANCE CHECK")
    print("=" * 50)
    
    model_types = ["random_forest", "naive_bayes", "logistic_regression"]
    
    for model_type in model_types:
        try:
            config = TrainingConfig(model_type=model_type)
            classifier = CelebrityMLClassifier(config)
            classifier.load_model(f"models/celebrity_classifier_{model_type}.pkl")
            
            print(f"\\n{model_type.upper()} MODEL:")
            print("-" * 30)
            
            if classifier.metrics:
                print(f"Accuracy: {classifier.metrics.accuracy:.3f}")
                print(f"Training Samples: {classifier.metrics.training_samples}")
                print(f"Test Samples: {classifier.metrics.test_samples}")
                
                # Show per-class F1 scores
                print("Per-class F1 scores:")
                for class_name, f1_score in classifier.metrics.f1_score.items():
                    if class_name not in ['macro avg', 'weighted avg']:
                        print(f"  {class_name}: {f1_score:.3f}")
            else:
                print("No metrics available")
                
        except FileNotFoundError:
            print(f"Model not found: {model_type}")
        except Exception as e:
            print(f"Error loading {model_type}: {e}")


async def main():
    """Main function running all examples."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Run prediction examples
        await predict_example()
        await batch_prediction_example()
        
        # Check model performance
        check_model_performance()
        
        print("\\n" + "=" * 50)
        print("All examples completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print("\\nTo use this example:")
        print("1. First run: python -m data_collection.train_classifier")
        print("2. Then run: python -m data_collection.predict_example")


if __name__ == "__main__":
    asyncio.run(main())