#!/usr/bin/env python3
"""
Training script for celebrity news classifier.
Collects data, prepares training samples, and trains ML models.
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.collection.config import DataCollectionConfig, EventCategory
from data.collection.models import NewsArticle, TrainingDataSample
from data.predictor.ml_classifier import CelebrityMLClassifier, TrainingConfig, TrainingDataManager
from data.predictor.pytorch_classifier import CelebrityPyTorchClassifier, PyTorchTrainingConfig
from data.predictor.model_evaluator import ModelEvaluator, evaluate_all_models
from data.collection.pipeline import CelebrityNewsDataPipeline
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import numpy as np

import logging
import argparse
import asyncio

logger = logging.getLogger(__name__)


class ClassifierTrainingPipeline:
    """Complete pipeline for training celebrity news classifier."""

    def __init__(self,
                 data_config: DataCollectionConfig = None,
                 training_config: TrainingConfig = None,
                 pytorch_config: PyTorchTrainingConfig = None):
        """Initialize training pipeline."""
        self.data_config = data_config or DataCollectionConfig()
        self.training_config = training_config or TrainingConfig()
        self.pytorch_config = pytorch_config or PyTorchTrainingConfig()
        self.data_pipeline = CelebrityNewsDataPipeline(self.data_config)
        self.classifier = CelebrityMLClassifier(self.training_config)
        self.pytorch_classifier = CelebrityPyTorchClassifier(self.pytorch_config)
        self.training_data_manager = TrainingDataManager()

    async def collect_training_data(self,
                                    max_articles: int = 1000,
                                    min_confidence: float = 0.6) -> List[TrainingDataSample]:
        """Collect and prepare training data."""
        logger.info(
            f"Collecting up to {max_articles} articles for training data")

        # Collect articles using existing pipeline
        articles, stats = await self.data_pipeline.run_collection()

        logger.info(f"Collected {len(articles)} articles")

        # Convert classified articles to training samples
        training_samples = self.training_data_manager.load_training_data_from_articles(
            articles, min_confidence=min_confidence
        )

        logger.info(f"Generated {len(training_samples)} training samples")

        # Show distribution
        from collections import Counter
        label_counts = Counter(
            sample.label.value for sample in training_samples)
        logger.info(f"Label distribution: {dict(label_counts)}")

        return training_samples

    def augment_training_data(self,
                              training_samples: List[TrainingDataSample]) -> List[TrainingDataSample]:
        """Augment training data with synthetic examples."""
        logger.info("Augmenting training data with synthetic examples")

        # Create synthetic examples for underrepresented categories
        from collections import Counter
        label_counts = Counter(
            sample.label.value for sample in training_samples)

        # Templates for synthetic data generation
        templates = {
            EventCategory.DIVORCE: [
                "{celebrity1} files for divorce from {celebrity2}",
                "{celebrity1} and {celebrity2} divorce after {years} years",
                "Celebrity couple {celebrity1} and {celebrity2} end marriage",
                "{celebrity1} confirms divorce proceedings with {celebrity2}"
            ],
            EventCategory.BREAKUP: [
                "{celebrity1} and {celebrity2} break up after {duration}",
                "{celebrity1} confirms split from {celebrity2}",
                "Celebrity couple {celebrity1} and {celebrity2} call it quits",
                "{celebrity1} and {celebrity2} end relationship"
            ],
            EventCategory.ENGAGEMENT: [
                "{celebrity1} announces engagement to {celebrity2}",
                "{celebrity1} and {celebrity2} get engaged",
                "Celebrity couple {celebrity1} and {celebrity2} engaged",
                "{celebrity1} proposes to {celebrity2}"
            ],
            EventCategory.MARRIAGE: [
                "{celebrity1} marries {celebrity2} in secret ceremony",
                "{celebrity1} and {celebrity2} tie the knot",
                "Celebrity wedding: {celebrity1} and {celebrity2}",
                "{celebrity1} and {celebrity2} exchange vows"
            ],
            EventCategory.PREGNANCY: [
                "{celebrity1} announces pregnancy",
                "{celebrity1} expecting first child with {celebrity2}",
                "{celebrity1} reveals baby news",
                "Celebrity {celebrity1} pregnant with second child"
            ],
            EventCategory.LAWSUIT: [
                "{celebrity1} sues {celebrity2} for defamation",
                "{celebrity1} files lawsuit against {celebrity2}",
                "Legal battle: {celebrity1} vs {celebrity2}",
                "{celebrity1} wins court case against {celebrity2}"
            ],
            EventCategory.FEUD: [
                "{celebrity1} feuds with {celebrity2} on social media",
                "{celebrity1} and {celebrity2} public feud escalates",
                "Celebrity drama: {celebrity1} vs {celebrity2}",
                "{celebrity1} calls out {celebrity2} publicly"
            ],
            EventCategory.SCANDAL: [
                "{celebrity1} involved in major scandal",
                "Scandal rocks {celebrity1}'s career",
                "{celebrity1} scandal causes public outrage",
                "Celebrity {celebrity1} under fire for scandal"
            ]
        }

        celebrity_names = [
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

        synthetic_samples = []
        target_samples_per_category = 20

        for category, count in label_counts.items():
            if count < target_samples_per_category:
                needed = target_samples_per_category - count
                category_enum = EventCategory(category)

                if category_enum in templates:
                    category_templates = templates[category_enum]

                    for i in range(needed):
                        template = category_templates[i % len(
                            category_templates)]

                        # Fill template with random celebrities
                        import random
                        celebrity1 = random.choice(celebrity_names)
                        celebrity2 = random.choice(
                            [name for name in celebrity_names if name != celebrity1])

                        text = template.format(
                            celebrity1=celebrity1,
                            celebrity2=celebrity2,
                            years=random.choice(["2", "3", "5", "10"]),
                            duration=random.choice(
                                ["6 months", "1 year", "2 years"])
                        )

                        sample = TrainingDataSample(
                            text=text,
                            label=category_enum,
                            source="synthetic",
                            confidence=0.8,
                            metadata={"generated": True}
                        )
                        synthetic_samples.append(sample)

        logger.info(
            f"Generated {len(synthetic_samples)} synthetic training samples")
        return training_samples + synthetic_samples

    async def train_and_evaluate(self,
                                 training_samples: List[TrainingDataSample],
                                 model_types: List[str] = None) -> dict:
        """Train and evaluate multiple model types including PyTorch models."""
        model_types = model_types or [
            "random_forest", "naive_bayes", "logistic_regression"]
        results = {}

        for model_type in model_types:
            logger.info(f"Training {model_type} classifier")

            if model_type in ["mlp", "lstm", "transformer"]:
                # PyTorch models
                config = PyTorchTrainingConfig(
                    model_type=model_type,
                    save_model=True,
                    model_save_path=f"{project_root}/models/celebrity_classifier_{model_type}.pth"
                )
                
                classifier = CelebrityPyTorchClassifier(config)
                metrics = classifier.train(training_samples)
                
                # Store results
                results[model_type] = {
                    'classifier': classifier,
                    'metrics': metrics,
                    'config': config,
                    'framework': 'pytorch'
                }
            else:
                # Scikit-learn models
                config = TrainingConfig(
                    model_type=model_type,
                    save_model=True,
                    model_save_path=f"{project_root}/models/celebrity_classifier_{model_type}.pkl",
                    perform_grid_search=False  # Can be enabled for better performance
                )

                classifier = CelebrityMLClassifier(config)
                metrics = classifier.train(training_samples)

                # Store results
                results[model_type] = {
                    'classifier': classifier,
                    'metrics': metrics,
                    'config': config,
                    'framework': 'sklearn'
                }

            logger.info(
                f"{model_type} training completed - Accuracy: {metrics.accuracy:.3f}")

        return results

    def compare_models(self, results: dict):
        """Compare performance of different models."""
        print("\\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)

        # Create comparison table
        comparison_data = []
        for model_type, result in results.items():
            metrics = result['metrics']
            framework = result.get('framework', 'sklearn')
            comparison_data.append({
                'Model': model_type,
                'Framework': framework,
                'Accuracy': f"{metrics.accuracy:.3f}",
                'CV Score': f"{np.mean(metrics.cross_val_scores):.3f} ± {np.std(metrics.cross_val_scores):.3f}",
                'Training Samples': metrics.training_samples,
                'Test Samples': metrics.test_samples
            })

        # Print comparison table
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))

        # Find best model
        best_model = max(
            results.keys(), key=lambda k: results[k]['metrics'].accuracy)
        print(f"\\nBest performing model: {best_model}")
        print(f"Best accuracy: {results[best_model]['metrics'].accuracy:.3f}")

        return best_model

    async def run_full_training_pipeline(self,
                                         max_articles: int = 1000,
                                         min_confidence: float = 0.6,
                                         augment_data: bool = True,
                                         model_types: List[str] = None,
                                         comprehensive_eval: bool = False) -> dict:
        """Run the complete training pipeline."""
        logger.info("Starting complete classifier training pipeline")

        # Step 1: Collect training data
        training_samples = await self.collect_training_data(max_articles, min_confidence)

        if len(training_samples) < 50:
            logger.warning(
                f"Only {len(training_samples)} training samples available. Consider collecting more data.")

        # Step 2: Augment data if requested
        if augment_data:
            training_samples = self.augment_training_data(training_samples)

        # Step 3: Save training data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_data_path = f"{project_root}/data/training_data_{timestamp}.json"
        Path(training_data_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.training_data_manager.save_training_data(
            training_samples, training_data_path)

        # Step 4: Train and evaluate models
        results = await self.train_and_evaluate(training_samples, model_types)

        # Step 5: Compare models and show results
        best_model = self.compare_models(results)

        # Step 6: Print detailed evaluation for best model
        print("\\n" + "=" * 80)
        print(f"DETAILED EVALUATION - BEST MODEL ({best_model})")
        print("=" * 80)
        results[best_model]['classifier'].print_evaluation_report()

        # Step 7: Comprehensive model evaluation and comparison (optional)
        if comprehensive_eval:
            print("\\n" + "=" * 80)
            print("COMPREHENSIVE MODEL EVALUATION")
            print("=" * 80)
            
            try:
                evaluator = evaluate_all_models(
                    training_results=results,
                    training_samples=training_samples,
                    output_dir=f"{project_root}/evaluation_results_{timestamp}"
                )
                
                # Get the statistically best model
                statistical_best_model, best_metrics = evaluator.get_best_model('f1_macro')
                print(f"\\nStatistically best model: {statistical_best_model}")
                print(f"F1 Score (Macro): {best_metrics.f1_macro:.4f}")
                print(f"Cross-validation: {best_metrics.cv_mean:.4f} ± {best_metrics.cv_std:.4f}")
                
            except Exception as e:
                logger.warning(f"Comprehensive evaluation failed: {e}")
                logger.info("Continuing with basic evaluation results")

        return results


async def main():
    """Main training function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Train celebrity news classifier")
    parser.add_argument("--max-articles", type=int, default=1000,
                        help="Maximum articles to collect for training")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="Minimum confidence threshold for training data")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable synthetic data augmentation")
    parser.add_argument("--models", nargs="+",
                        choices=["random_forest", "naive_bayes",
                                 "logistic_regression", "svm", 
                                 "mlp", "lstm", "transformer"],
                        default=["random_forest", "naive_bayes",
                                 "logistic_regression"],
                        help="Models to train and compare (includes PyTorch: mlp, lstm, transformer)")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save trained models")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--comprehensive-eval", action="store_true",
                        help="Enable comprehensive model evaluation with statistical testing")

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set up training configs
    data_config = DataCollectionConfig(
        max_articles_per_run=args.max_articles,
        output_directory=f"{project_root}/data/training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    training_config = TrainingConfig(
        model_save_path=f"{args.output_dir}/celebrity_classifier.pkl"
    )

    # Run training pipeline
    pipeline = ClassifierTrainingPipeline(data_config, training_config)

    try:
        results = await pipeline.run_full_training_pipeline(
            max_articles=args.max_articles,
            min_confidence=args.min_confidence,
            augment_data=not args.no_augment,
            model_types=args.models,
            comprehensive_eval=args.comprehensive_eval
        )

        print("\\n" + "=" * 80)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Trained {len(results)} models")
        print(f"Models saved to: {args.output_dir}/")
        print("\\nYou can now use the trained classifier to predict celebrity news events.")

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
