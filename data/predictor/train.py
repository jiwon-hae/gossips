import argparse
import logging
import asyncio
import os

from datetime import datetime
from train_classifier import ClassifierTrainingPipeline, TrainingConfig, DataCollectionConfig
from pathlib import Path

logger = logging.getLogger(__name__)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   

async def main():
    """Main training function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Train celebrity news classifier")
    parser.add_argument("--max-articles", type=int, default=1000,
                        help="Maximum articles to collect for training")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="Minimum confidence threshold for training data")
    parser.add_argument("--no-augment", action="store_true", default = True,
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
    parser.add_argument("--comprehensive-eval", action="store_true", help="Enable comprehensive_eval")

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create output directory
    output_dir = f'{project_root}/{args.output_dir}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set up training configs
    data_config = DataCollectionConfig(
        max_articles_per_run=args.max_articles,
        output_directory=f"{project_root}/data/training"
    )

    training_config = TrainingConfig(
        model_save_path=f"{output_dir}/celebrity_classifier.pkl"
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
        print(f"Models saved to: {output_dir}/")
        print("\\nYou can now use the trained classifier to predict celebrity news events.")

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
