#!/usr/bin/env python
"""NIDS training CLI (PyTorch Lightning)."""

import argparse
import sys
from pathlib import Path

import mlflow
import property_driven_ml as pdml
import torch
from loguru import logger

from src.config import config
from src.data import DataLoader
from src.models import create_model
from src.training import NIDSLightningModule, create_trainer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NIDS Training with PyTorch Lightning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default=config.model.model_type,
        choices=config.model.model_types,
        help="Model architecture",
    )
    parser.add_argument(
        "--training-type",
        type=str,
        default="base",
        choices=["base", "adversarial", "constraint", "vehicle"],
        help="Type of training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.training.epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=config.training.steps_per_epoch,
        help="Limit training steps per epoch (0 = full epoch)",
    )
    parser.add_argument(
        "--attack-pattern",
        type=str,
        default="hulk",
        choices=["hulk", "goodHTTP1", "goodHTTP2", "slowIATsAttacks", "invalid", "mixed"],
        help="Attack pattern for adversarial training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (auto = detect automatically)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fixed",
        choices=["original", "fixed"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=config.logging.level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main() -> int:
    """Main training entry point."""
    try:
        args = parse_arguments()

        log_dir = Path(config.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.remove()
        logger.add(
            sys.stderr,
            format="<level>{time:YYYY-MM-DD HH:mm:ss}</level> | <level>{level: <8}</level> | <level>{message}</level>",
            level=args.log_level,
            colorize=True,
        )
        logger.add(
            log_dir / "nids_training.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=args.log_level,
            rotation="500 MB",
            retention=5,
        )

        # Setup MLflow experiment tracking
        if config.mlflow.enabled:
            try:
                if config.mlflow.tracking_uri:
                    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

                experiment_name = args.experiment_name or config.mlflow.experiment_name
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(
                        experiment_name, artifact_location=config.mlflow.artifact_location
                    )
                    logger.info(f"Created MLflow experiment: {experiment_name}")
                else:
                    logger.info(f"Using existing MLflow experiment: {experiment_name}")

                mlflow.set_experiment(experiment_name)
                mlflow.pytorch.autolog()  # type: ignore[attr-defined]
            except Exception as e:
                logger.error(f"Failed to setup MLflow: {e}")

        logger.info("Starting NIDS training pipeline")
        logger.info(
            f"Configuration: model_type={args.model_type}, training_type={args.training_type}"
        )

        logger.info("Loading data...")
        data_loader = DataLoader(data_dir=config.data.data_dir)
        train_loader, val_loader, test_loader, feature_names = data_loader.load_data(
            dataset=args.dataset
        )
        input_size = len(feature_names)
        logger.info(f"Data loaded: {input_size} features, {len(train_loader)} train batches")

        logger.info(f"Creating {args.model_type} model...")
        model = create_model(input_size, args.model_type)
        logger.info("Model created successfully")

        accelerator = args.device

        module_kwargs = {
            "model": model,
            "training_type": args.training_type,
        }

        if args.training_type == "adversarial":
            module_kwargs["attack_pattern"] = args.attack_pattern
            logger.info("Adversarial training with research hyperrectangles")
        elif args.training_type == "constraint":
            device = torch.device(
                "cuda" if accelerator == "cuda" and torch.cuda.is_available() else "cpu"
            )
            constraint = pdml.constraints.StandardRobustnessConstraint(
                device=device,
                epsilon=config.training.pgd_epsilon,
            )
            module_kwargs["constraint"] = constraint
            logger.info("Property-driven ML constraint training")
        elif args.training_type == "vehicle":
            logger.info("Vehicle property-driven training")
        else:
            logger.info("Baseline training")

        module = NIDSLightningModule(**module_kwargs)

        trainer = create_trainer(
            training_type=args.training_type,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            experiment_name=args.experiment_name,
        )

        # Log configuration parameters to MLflow
        if config.mlflow.enabled and mlflow.active_run():
            mlflow.log_params(
                {
                    "model.type": config.model.model_type,
                    "training.epochs": config.training.epochs,
                    "training.learning_rate": config.training.learning_rate,
                    "training.steps_per_epoch": config.training.steps_per_epoch,
                    "training.pgd_epsilon": config.training.pgd_epsilon,
                    "data.batch_size": config.data.batch_size,
                }
            )

        logger.info("Starting training...")
        trainer.fit(
            module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        logger.info("Evaluating on test set...")
        test_results = trainer.test(module, dataloaders=test_loader)
        if test_results:
            test_loss = test_results[0].get("test_loss", 0.0)
            test_acc = test_results[0].get("test_accuracy", 0.0)
            logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        logger.info("Training completed successfully!")
        logger.info("Results and model artifacts logged to MLflow")
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
