"""
Main entry point for NIDS training
"""

import argparse
import os
import sys

# Configure TensorFlow before importing other modules
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Suppress INFO messages
import tensorflow as tf

from src.config import config
from src.data import DataLoader
from src.models import create_model
from src.training import train_adversarial, train_base
from src.utils.logging import get_logger, setup_logging
from src.utils.performance import configure_tensorflow_performance, log_system_info


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
        choices=["adversarial", "base"],
        help="Type of training (adversarial or base)",
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
        default="preprocessed",
        choices=["preprocessed", "cic_wednesday"],
        help="Dataset to use (preprocessed: original researcher data ~108k samples, cic_wednesday: larger dataset ~799k samples)",
    )

    return parser.parse_args()


def setup_tensorflow(device: str = "auto") -> None:
    """Setup TensorFlow configuration"""
    logger = get_logger(__name__)

    # Configure performance optimizations
    configure_tensorflow_performance()

    # Device configuration
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        logger.info("Forcing CPU usage")
    elif device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s) with memory growth")
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {e}")
        else:
            logger.warning("No GPUs found, falling back to CPU")
    else:  # auto
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Auto-configured {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"GPU auto-configuration failed: {e}")
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

        # Setup TensorFlow
        setup_tensorflow(args.device)

        # Log system information
        log_system_info()

        # Load data
        logger.info("Loading data...")
        data_loader = DataLoader(data_dir=config.data.data_dir)
        train_dataset, val_dataset, test_dataset, feature_names = data_loader.load_data(
            dataset=args.dataset
        )
        input_size = len(feature_names)
        logger.info(f"Loaded data with {input_size} features")

        # Create model
        logger.info(f"Creating {args.model_type} model...")
        model = create_model(input_size, args.model_type)

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        logger.info("Model compiled successfully")
        if logger.logger.level <= 20:  # INFO level or below
            model.summary()

        # Training
        if args.training_type == "adversarial":
            logger.info("Starting adversarial training...")

            # Use research-based hyperrectangles instead of generic attack rectangles
            logger.info("Using research-based hyperrectangle definitions from original paper")

            # The training module should handle research hyperrectangles internally
            # No need to pass attack_rects explicitly - they're built into the research
            history = train_adversarial(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                attack_rects=None,  # Let training use research hyperrectangles
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
            )

        else:  # base training
            logger.info("Starting base training...")

            # Train without adversarial examples
            history = train_base(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
            )

        # Final evaluation
        logger.info("Evaluating on test set...")
        test_results = model.evaluate(test_dataset, verbose=0)
        logger.info(f"Test Loss: {test_results[0]:.4f}, Test Accuracy: {test_results[1]:.4f}")

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
