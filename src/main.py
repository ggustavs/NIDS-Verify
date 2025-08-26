"""
Main entry point for NIDS training (PyTorch)
"""

import argparse
import os
import sys
from typing import cast

import torch
from property_driven_ml.utils.factories import CreateEpsilonBall

from src.config import config
from src.data import DataLoader, _NDArrayDataset
from src.models import create_model
from src.training import train_adversarial, train_base, train_constraint
from src.utils.logging import get_logger, setup_logging
from src.utils.performance import configure_torch_performance, log_system_info


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="NIDS Adversarial Training")

    parser.add_argument(
        "--model-type",
        type=str,
        default=config.model.model_type,
        choices=["small", "mid", "mid2", "mid3", "mid4", "big", "big2", "big3", "big4", "massive"],
        help="Model architecture to use",
    )

    parser.add_argument(
        "--training-type",
        type=str,
        default="adversarial",
        choices=["adversarial", "constraint", "base"],
        help="Type of training (adversarial (with default hyperrectangles), constraint (with custom hyperrectangles and constraint loss), or base)",
    )

    parser.add_argument(
        "--epochs", type=int, default=config.training.epochs, help="Number of training epochs"
    )

    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=config.training.steps_per_epoch,
        help="Number of steps per epoch",
    )

    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name")

    parser.add_argument(
        "--log-level",
        type=str,
        default=config.logging.level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--attack-pattern",
        type=str,
        default="hulk",
        choices=["hulk", "goodHTTP1", "goodHTTP2", "slowIATsAttacks", "invalid", "mixed"],
        help="Research attack pattern to use for adversarial training",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Device to use for training",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="fixed",
        choices=["original", "fixed"],
        help="Dataset to use (original: preprocessed-dos-*, fixed: CICWednesday pos_neg split)",
    )

    return parser.parse_args()


def setup_device(device: str = "auto") -> None:
    """Setup PyTorch device configuration"""
    logger = get_logger(__name__)
    configure_torch_performance()

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Forcing CPU usage")
    elif device == "gpu":
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("No GPUs found, falling back to CPU")
    else:
        if torch.cuda.is_available():
            logger.info(f"Auto-selected GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("No GPUs found, using CPU")


def main() -> int:
    """Main training function"""
    logger = None
    try:
        # Parse arguments
        args = parse_arguments()

        # Setup logging
        setup_logging(level=args.log_level, experiment_name=args.experiment_name)
        logger = get_logger(__name__)

        logger.info("Starting NIDS training pipeline")
        logger.info(f"Arguments: {vars(args)}")

        # Setup device (PyTorch)
        setup_device(args.device)

        # Log system information
        log_system_info()

        # Load data
        logger.info("Loading data...")
        data_loader = DataLoader(data_dir=config.data.data_dir)
        train_loader, val_loader, test_loader, feature_names = data_loader.load_data(
            dataset=args.dataset
        )
        input_size = len(feature_names)
        if args.training_type == "constraint":
            logger.info("Making constraint training compliant DataLoader...")
            train_factory, _ = CreateEpsilonBall(args.epsilon)
            train_ds: _NDArrayDataset = cast(_NDArrayDataset, train_loader.dataset)
            wrapper_train = train_factory(train_ds, train_ds.mean, train_ds.std)

        logger.info(f"Loaded data with {input_size} features")

        # Create model
        logger.info(f"Creating {args.model_type} model...")
        model = create_model(input_size, args.model_type)
        logger.info("Model created successfully")

        # Training
        if args.training_type == "adversarial":
            logger.info("Starting adversarial training...")

            # Use research-based hyperrectangles instead of generic attack rectangles
            logger.info("Using research-based hyperrectangle definitions from original paper")

            # The training module should handle research hyperrectangles internally
            # No need to pass attack_rects explicitly - they're built into the research
            history = train_adversarial(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                attack_rects=None,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                attack_pattern=args.attack_pattern,
            )
        elif args.training_type == "constraint":
            logger.info("Starting constraint-based adversarial training...")

            history = train_constraint(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                attack_rects=wrapper_train,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                attack_pattern=args.attack_pattern,
            )
        else:  # base training
            logger.info("Starting base training...")

            # Train without adversarial examples
            history = train_base(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
            )

        # Final evaluation (Torch)
        logger.info("Evaluating on test set...")
        from src.training.trainer import evaluate_model as torch_evaluate

        test_loss, test_acc = torch_evaluate(model, test_loader)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # Log training summary
        final_train_acc = history["accuracy"][-1] if history["accuracy"] else 0
        final_val_acc = history["val_accuracy"][-1] if history["val_accuracy"] else 0
        avg_epoch_time = (
            sum(history["epoch_times"]) / len(history["epoch_times"])
            if history["epoch_times"]
            else 0
        )

        logger.info("Training completed successfully!")
        logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
        logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
        logger.info(f"Average Epoch Time: {avg_epoch_time:.1f}s")
        logger.info("Model saved and registered in MLflow (see logs above for details)")

        return 0

    except KeyboardInterrupt:
        if logger:
            logger.info("Training interrupted by user")
        else:
            print("Training interrupted by user")
        return 1
    except Exception as e:
        if logger:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
        else:
            print(f"Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
