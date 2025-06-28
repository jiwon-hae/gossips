#!/usr/bin/env python3
"""
Machine Learning classifier for celebrity news events.
Uses scikit-learn for training and prediction.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import Counter

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    import joblib
except ImportError:
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")

from data.collection.models import NewsArticle, TrainingDataSample, ClassificationResult
from data.collection.config import EventCategory, DataCollectionConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for model evaluation."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    support: Dict[str, int]
    confusion_matrix: np.ndarray
    classification_report: str
    cross_val_scores: List[float]
    training_samples: int
    test_samples: int


@dataclass
class TrainingConfig:
    """Configuration for ML training."""
    model_type: str = "random_forest"  # random_forest, naive_bayes, svm, logistic_regression
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    max_features: int = 5000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: Tuple[int, int] = (1, 2)  # Unigrams and bigrams
    perform_grid_search: bool = False
    save_model: bool = True
    model_save_path: str = f"{project_root}/models/celebrity_classifier.pkl"


class CelebrityMLClassifier:
    """Machine Learning classifier for celebrity news events."""
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize the ML classifier."""
        self.config = config or TrainingConfig()
        self.pipeline: Optional[Pipeline] = None
        self.label_encoder = LabelEncoder()
        self.metrics: Optional[ModelMetrics] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        self.is_trained = False
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
        
    def _get_model(self) -> Any:
        """Get the specified model."""
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            "naive_bayes": MultinomialNB(),
            "svm": SVC(
                random_state=self.config.random_state,
                probability=True
            ),
            "logistic_regression": LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            )
        }
        
        if self.config.model_type not in models:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        return models[self.config.model_type]
    
    def _prepare_training_data(self, training_samples: List[TrainingDataSample]) -> Tuple[List[str], List[str]]:
        """Prepare training data from samples."""
        texts = []
        labels = []
        
        for sample in training_samples:
            texts.append(sample.text)
            labels.append(sample.label.value)
        
        return texts, labels
    
    def _create_pipeline(self) -> Pipeline:
        """Create ML pipeline with text preprocessing and model."""
        vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            ngram_range=self.config.ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        
        model = self._get_model()
        
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', model)
        ])
        
        return pipeline
    
    def _perform_grid_search(self, X: List[str], y: List[str]) -> Pipeline:
        """Perform hyperparameter tuning with grid search."""
        logger.info("Performing grid search for hyperparameter tuning...")
        
        # Define parameter grids for different models
        param_grids = {
            "random_forest": {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'tfidf__max_features': [3000, 5000, 10000]
            },
            "naive_bayes": {
                'classifier__alpha': [0.1, 1.0, 10.0],
                'tfidf__max_features': [3000, 5000, 10000]
            },
            "logistic_regression": {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga'],
                'tfidf__max_features': [3000, 5000, 10000]
            }
        }
        
        if self.config.model_type in param_grids:
            param_grid = param_grids[self.config.model_type]
            
            pipeline = self._create_pipeline()
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            
            return grid_search.best_estimator_
        else:
            logger.warning(f"Grid search not implemented for {self.config.model_type}")
            return self._create_pipeline()
    
    def train(self, training_samples: List[TrainingDataSample]) -> ModelMetrics:
        """Train the classifier on training samples."""
        logger.info(f"Training {self.config.model_type} classifier on {len(training_samples)} samples")
        
        # Prepare data
        texts, labels = self._prepare_training_data(training_samples)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Check class distribution
        class_counts = Counter(labels)
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        # Check if stratification is possible
        min_class_count = min(class_counts.values())
        use_stratify = min_class_count >= 2
        
        if not use_stratify:
            logger.warning(f"Some classes have only {min_class_count} sample(s). Disabling stratification.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y_encoded, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_encoded if use_stratify else None
        )
        
        # Create and train pipeline
        if self.config.perform_grid_search:
            self.pipeline = self._perform_grid_search(X_train, y_train)
        else:
            self.pipeline = self._create_pipeline()
            self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Convert encoded labels back to original labels for reporting
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Classification report
        report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        report_str = classification_report(y_test_labels, y_pred_labels)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.pipeline, texts, y_encoded, 
            cv=self.config.cross_validation_folds,
            scoring='f1_weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Extract metrics
        precision = {k: v['precision'] for k, v in report.items() if isinstance(v, dict)}
        recall = {k: v['recall'] for k, v in report.items() if isinstance(v, dict)}
        f1 = {k: v['f1-score'] for k, v in report.items() if isinstance(v, dict)}
        support = {k: v['support'] for k, v in report.items() if isinstance(v, dict)}
        
        # Create metrics object
        self.metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            support=support,
            confusion_matrix=cm,
            classification_report=report_str,
            cross_val_scores=cv_scores.tolist(),
            training_samples=len(X_train),
            test_samples=len(X_test)
        )
        
        # Extract feature importance (for tree-based models)
        self._extract_feature_importance()
        
        self.is_trained = True
        
        # Save model if configured
        if self.config.save_model:
            self.save_model()
        
        logger.info(f"Training completed. Accuracy: {accuracy:.3f}, CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return self.metrics
    
    def _extract_feature_importance(self):
        """Extract feature importance from trained model."""
        if not self.pipeline or not hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            return
        
        try:
            # Get feature names and importances
            feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
            importances = self.pipeline.named_steps['classifier'].feature_importances_
            
            # Create feature importance dictionary
            self.feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
    
    def predict(self, texts: List[str]) -> List[ClassificationResult]:
        """Predict labels for new texts."""
        if not self.is_trained or not self.pipeline:
            raise ValueError("Model must be trained before making predictions")
        
        # Make predictions
        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)
        
        results = []
        for i, text in enumerate(texts):
            # Get predicted class and confidence
            predicted_class_encoded = predictions[i]
            predicted_class = self.label_encoder.inverse_transform([predicted_class_encoded])[0]
            confidence = probabilities[i].max()
            
            # Get all class probabilities
            all_probs = dict(zip(
                self.label_encoder.inverse_transform(range(len(probabilities[i]))),
                probabilities[i]
            ))
            
            # Convert to EventCategory
            try:
                category = EventCategory(predicted_class)
            except ValueError:
                category = EventCategory.OTHER
            
            result = ClassificationResult(
                category=category,
                confidence=confidence,
                method="ml_classifier",
                raw_scores={EventCategory(k): float(v) for k, v in all_probs.items() if k in [e.value for e in EventCategory]},
                explanation=f"ML prediction with {self.config.model_type}"
            )
            
            results.append(result)
        
        return results
    
    def predict_article(self, article: NewsArticle) -> ClassificationResult:
        """Predict category for a single article."""
        text = article.title
        if article.summary:
            text += " " + article.summary
        
        results = self.predict([text])
        return results[0]
    
    def save_model(self, path: str = None):
        """Save the trained model and metadata."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        save_path = path or self.config.model_save_path
        
        # Save the complete model state
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'trained_at': datetime.now()
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str = None):
        """Load a previously trained model."""
        load_path = path or self.config.model_save_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        model_data = joblib.load(load_path)
        
        self.pipeline = model_data['pipeline']
        self.label_encoder = model_data['label_encoder']
        self.metrics = model_data.get('metrics')
        self.feature_importance = model_data.get('feature_importance')
        self.is_trained = True
        
        logger.info(f"Model loaded from {load_path}")
        logger.info(f"Model trained at: {model_data.get('trained_at', 'Unknown')}")
    
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if not self.feature_importance:
            return []
        
        return list(self.feature_importance.items())[:n]
    
    def print_evaluation_report(self):
        """Print detailed evaluation report."""
        if not self.metrics:
            print("No evaluation metrics available. Train the model first.")
            return
        
        print("=" * 60)
        print("CELEBRITY NEWS CLASSIFIER - EVALUATION REPORT")
        print("=" * 60)
        
        print(f"Model Type: {self.config.model_type}")
        print(f"Training Samples: {self.metrics.training_samples}")
        print(f"Test Samples: {self.metrics.test_samples}")
        print(f"Overall Accuracy: {self.metrics.accuracy:.3f}")
        print(f"Cross-Validation Score: {np.mean(self.metrics.cross_val_scores):.3f} ± {np.std(self.metrics.cross_val_scores):.3f}")
        
        print("\nPer-Class Metrics:")
        print("-" * 40)
        for class_name in self.metrics.precision.keys():
            if class_name in ['macro avg', 'weighted avg']:
                continue
            precision = self.metrics.precision.get(class_name, 0)
            recall = self.metrics.recall.get(class_name, 0)
            f1 = self.metrics.f1_score.get(class_name, 0)
            support = self.metrics.support.get(class_name, 0)
            print(f"{class_name:15}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} Support={support}")
        
        if self.feature_importance:
            print(f"\nTop 10 Most Important Features:")
            print("-" * 40)
            for feature, importance in self.get_top_features(10):
                print(f"{feature:25}: {importance:.4f}")
        
        print("\nDetailed Classification Report:")
        print("-" * 40)
        print(self.metrics.classification_report)


# Training utilities
class TrainingDataManager:
    """Manager for handling training data operations."""
    
    @staticmethod
    def load_training_data_from_articles(articles: List[NewsArticle], 
                                       min_confidence: float = 0.6) -> List[TrainingDataSample]:
        """Convert classified articles to training samples."""
        training_samples = []
        
        for article in articles:
            if (article.classified and 
                article.classification_confidence and 
                article.classification_confidence >= min_confidence and
                article.predicted_category and
                article.predicted_category != EventCategory.OTHER):
                
                sample = TrainingDataSample(
                    text=article.title,
                    label=article.predicted_category,
                    source=article.source,
                    confidence=article.classification_confidence,
                    metadata={
                        'url': article.url,
                        'published_date': article.published_date.isoformat(),
                        'keywords_matched': article.keywords_matched
                    }
                )
                training_samples.append(sample)
        
        return training_samples
    
    @staticmethod
    def save_training_data(samples: List[TrainingDataSample], path: str):
        """Save training data to file."""
        data = [sample.to_dict() for sample in samples]
        
        if path.endswith('.json'):
            import json
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif path.endswith('.csv'):
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")
        
        logger.info(f"Training data saved to {path}")
    
    @staticmethod
    def load_training_data(path: str) -> List[TrainingDataSample]:
        """Load training data from file."""
        if path.endswith('.json'):
            import json
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
            data = df.to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")
        
        samples = []
        for item in data:
            sample = TrainingDataSample(
                text=item['text'],
                label=EventCategory(item['label']),
                source=item['source'],
                confidence=item.get('confidence', 1.0),
                metadata=item.get('metadata', {})
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} training samples from {path}")
        return samples


# Example usage and testing
async def train_example_classifier():
    """Example function showing how to train the classifier."""
    print("Celebrity News ML Classifier Training Example")
    print("=" * 50)
    
    # Create sample training data (in practice, this would come from your data collection)
    sample_data = [
        ("Taylor Swift and Joe Alwyn break up after 6 years", EventCategory.BREAKUP),
        ("Kim Kardashian files for divorce from Pete Davidson", EventCategory.DIVORCE),
        ("Jennifer Lopez announces engagement to Ben Affleck", EventCategory.ENGAGEMENT),
        ("Brad Pitt and Angelina Jolie custody battle intensifies", EventCategory.CUSTODY_BATTLE),
        ("Johnny Depp wins defamation lawsuit against Amber Heard", EventCategory.LAWSUIT),
        ("Britney Spears reveals pregnancy with fourth child", EventCategory.PREGNANCY),
        ("Kanye West starts controversial Twitter feud with Drake", EventCategory.FEUD),
        ("Selena Gomez opens up about mental health in interview", EventCategory.HEALTH_ISSUES),
        ("Leonardo DiCaprio spotted with new mystery woman", EventCategory.DATING),
        ("Rihanna launches new beauty business line", EventCategory.BUSINESS_VENTURE),
    ]
    
    # Create training samples
    training_samples = []
    for text, label in sample_data:
        sample = TrainingDataSample(
            text=text,
            label=label,
            source="example",
            confidence=1.0
        )
        training_samples.append(sample)
    
    # Add more diverse samples for each category
    for category in EventCategory:
        if category != EventCategory.OTHER:
            for i in range(5):  # Add 5 examples per category
                sample = TrainingDataSample(
                    text=f"Celebrity news about {category.value} event {i+1}",
                    label=category,
                    source="synthetic",
                    confidence=0.8
                )
                training_samples.append(sample)
    
    print(f"Created {len(training_samples)} training samples")
    
    # Train different models
    model_types = ["random_forest", "naive_bayes", "logistic_regression"]
    
    for model_type in model_types:
        print(f"\nTraining {model_type} classifier...")
        
        config = TrainingConfig(
            model_type=model_type,
            save_model=True,
            model_save_path=f"{project_root}/models/celebrity_classifier_{model_type}.pkl"
        )
        
        classifier = CelebrityMLClassifier(config)
        metrics = classifier.train(training_samples)
        
        classifier.print_evaluation_report()
        
        # Test prediction
        test_headlines = [
            "Actress announces surprise divorce from husband",
            "Singer spotted with new romantic partner",
            "Celebrity couple expecting their first child"
        ]
        
        print("\nTesting predictions:")
        results = classifier.predict(test_headlines)
        for headline, result in zip(test_headlines, results):
            print(f"'{headline}' -> {result.category.value} (confidence: {result.confidence:.3f})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(train_example_classifier())