# Celebrity News ML Classifier

A machine learning system for automatically classifying celebrity news articles into event categories like divorce, engagement, lawsuits, etc.

## Overview

This ML classifier uses scikit-learn to train models that can automatically categorize celebrity news headlines and articles into predefined event types. It includes:

- **Data Collection**: Automatically collects celebrity news from Google News
- **Training Pipeline**: Converts classified articles into training data
- **ML Models**: Multiple model types (Random Forest, Naive Bayes, Logistic Regression, SVM)
- **Evaluation**: Comprehensive model evaluation and comparison
- **Prediction**: Easy-to-use prediction interface for new articles

## Event Categories

The classifier can identify these celebrity event types:

- **Relationship Events**: Divorce, Breakup, Engagement, Marriage, Dating, Reconciliation
- **Family Events**: Pregnancy, Birth, Adoption, Custody Battle
- **Professional Events**: Career Move, Business Venture, Collaboration
- **Legal Events**: Lawsuit, Arrest, Court Case
- **Social Events**: Feud, Scandal, Controversy, Public Dispute
- **Health Events**: Health Issues, Hospitalization, Recovery
- **Other Events**: Award, Achievement, Death, Retirement

## Quick Start

### 1. Install Dependencies

```bash
pip install scikit-learn pandas numpy
```

Optional dependencies:
```bash
pip install pyarrow  # for parquet export
pip install matplotlib seaborn  # for visualization
```

### 2. Train the Classifier

```bash
# Basic training with default settings
python -m data_collection.train_classifier

# Advanced training with custom parameters
python -m data_collection.train_classifier \\
    --max-articles 2000 \\
    --min-confidence 0.7 \\
    --models random_forest logistic_regression \\
    --verbose
```

### 3. Use the Trained Model

```python
from data_collection.ml_classifier import CelebrityMLClassifier, TrainingConfig
from data_collection.models import NewsArticle
from datetime import datetime

# Load trained model
config = TrainingConfig(model_type="random_forest")
classifier = CelebrityMLClassifier(config)
classifier.load_model("models/celebrity_classifier_random_forest.pkl")

# Classify a headline
headline = "Taylor Swift and Travis Kelce break up after year-long romance"
results = classifier.predict([headline])

print(f"Category: {results[0].category.value}")
print(f"Confidence: {results[0].confidence:.3f}")

# Classify a full article
article = NewsArticle(
    title="Celebrity Couple Announces Divorce",
    url="https://example.com/news",
    source="Entertainment News",
    published_date=datetime.now(),
    summary="After 5 years of marriage, the couple has decided to separate."
)

result = classifier.predict_article(article)
print(f"Predicted: {result.category.value} (confidence: {result.confidence:.3f})")
```

## Training Configuration

### TrainingConfig Options

```python
from data_collection.ml_classifier import TrainingConfig

config = TrainingConfig(
    model_type="random_forest",          # Model type to train
    test_size=0.2,                       # Test set size (20%)
    random_state=42,                     # For reproducibility
    cross_validation_folds=5,            # K-fold CV
    max_features=5000,                   # Maximum TF-IDF features
    min_df=2,                           # Minimum document frequency
    max_df=0.95,                        # Maximum document frequency
    ngram_range=(1, 2),                 # Unigrams and bigrams
    perform_grid_search=False,          # Enable hyperparameter tuning
    save_model=True,                    # Save trained model
    model_save_path="models/classifier.pkl"
)
```

### Available Model Types

1. **Random Forest** (`random_forest`)
   - Good overall performance
   - Handles feature interactions well
   - Provides feature importance

2. **Naive Bayes** (`naive_bayes`)
   - Fast training and prediction
   - Works well with text data
   - Good baseline model

3. **Logistic Regression** (`logistic_regression`)
   - Interpretable coefficients
   - Fast and stable
   - Good for linearly separable classes

4. **Support Vector Machine** (`svm`)
   - Good for high-dimensional data
   - Effective with limited training data
   - Slower training time

## Training Pipeline

### 1. Data Collection and Preparation

```python
from data_collection.train_classifier import ClassifierTrainingPipeline

pipeline = ClassifierTrainingPipeline()

# Collect training data from news sources
training_samples = await pipeline.collect_training_data(
    max_articles=1000,
    min_confidence=0.6
)

# Augment with synthetic examples
training_samples = pipeline.augment_training_data(training_samples)
```

### 2. Model Training and Evaluation

```python
# Train multiple models
results = await pipeline.train_and_evaluate(
    training_samples,
    model_types=["random_forest", "naive_bayes", "logistic_regression"]
)

# Compare model performance
best_model = pipeline.compare_models(results)
```

### 3. Model Evaluation

The training pipeline provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **Cross-Validation**: K-fold cross-validation scores
- **Confusion Matrix**: Detailed error analysis
- **Feature Importance**: Most important words/features

## Usage Examples

### Command Line Training

```bash
# Basic training
python -m data_collection.train_classifier

# Custom configuration
python -m data_collection.train_classifier \\
    --max-articles 1500 \\
    --min-confidence 0.65 \\
    --models random_forest logistic_regression \\
    --output-dir my_models \\
    --verbose

# Without synthetic data augmentation
python -m data_collection.train_classifier --no-augment
```

### Prediction Examples

```bash
# Run prediction examples
python -m data_collection.predict_example
```

### Batch Prediction

```python
# Classify multiple articles at once
headlines = [
    "Celebrity couple announces engagement",
    "Actor files lawsuit against tabloid",
    "Singer reveals pregnancy news"
]

results = classifier.predict(headlines)
for headline, result in zip(headlines, results):
    print(f"{headline} -> {result.category.value} ({result.confidence:.3f})")
```

## Model Performance

### Typical Performance Metrics

With sufficient training data (1000+ samples), you can expect:

- **Random Forest**: 80-90% accuracy
- **Logistic Regression**: 75-85% accuracy  
- **Naive Bayes**: 70-80% accuracy
- **SVM**: 75-85% accuracy

Performance depends on:
- Quality and quantity of training data
- Balance of event categories
- Clarity of celebrity news headlines
- Keyword overlap between categories

### Improving Performance

1. **More Training Data**: Collect more articles for underrepresented categories
2. **Better Keywords**: Refine keyword lists for each event category
3. **Feature Engineering**: Add custom features like named entity recognition
4. **Hyperparameter Tuning**: Enable grid search for optimal parameters
5. **Ensemble Methods**: Combine multiple models for better accuracy

## File Structure

```
data_collection/
├── ml_classifier.py           # Main ML classifier implementation
├── train_classifier.py       # Training pipeline and CLI
├── predict_example.py         # Usage examples
├── models.py                  # Data models
├── config.py                  # Configuration classes
└── models/                    # Saved model files
    ├── celebrity_classifier_random_forest.pkl
    ├── celebrity_classifier_naive_bayes.pkl
    └── celebrity_classifier_logistic_regression.pkl
```

## Advanced Usage

### Custom Feature Engineering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Custom TF-IDF configuration
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),         # Include trigrams
    min_df=3,
    max_df=0.9,
    stop_words='english',
    analyzer='word'
)

# Create custom pipeline
pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('classifier', RandomForestClassifier(n_estimators=200))
])
```

### Grid Search for Hyperparameter Tuning

```python
config = TrainingConfig(
    model_type="random_forest",
    perform_grid_search=True     # Enable grid search
)

# This will automatically find best hyperparameters
classifier = CelebrityMLClassifier(config)
metrics = classifier.train(training_samples)
```

### Loading and Saving Models

```python
# Save model
classifier.save_model("my_custom_model.pkl")

# Load model
classifier.load_model("my_custom_model.pkl")

# Check if model is trained
if classifier.is_trained:
    results = classifier.predict(["Some celebrity news headline"])
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'sklearn'**
   ```bash
   pip install scikit-learn
   ```

2. **Low accuracy (< 60%)**
   - Collect more training data
   - Check keyword quality in config.py
   - Try different model types
   - Enable data augmentation

3. **FileNotFoundError: Model file not found**
   - Run training script first
   - Check model save path
   - Verify model_type matches saved model

4. **Imbalanced classes warning**
   - Enable data augmentation
   - Collect more data for underrepresented categories
   - Use stratified sampling

### Performance Tips

1. **Use Random Forest** for best overall performance
2. **Enable data augmentation** for better category coverage
3. **Increase max_articles** for more training data
4. **Tune min_confidence** threshold based on your quality requirements
5. **Regular retraining** with new data improves performance

## Contributing

To add new event categories:

1. Update `EventCategory` enum in `config.py`
2. Add keywords for the new category
3. Update synthetic data templates in training pipeline
4. Retrain models with new category data

## License

This project is part of the RAG with LlamaIndex system and follows the same license terms.