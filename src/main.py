#!/usr/bin/env python
"""NIDS training CLI (PyTorch Lightning + MLflow)."""

import argparse
import sys
from pathlib import Path

import mlflow
import torch
from loguru import logger

from src.config import config
from src.data import NIDSDataModule
from src.models import MODEL_TYPES, create_model
from src.training import NIDSLightningModule, create_trainer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIDS training")
    parser.add_argument(
        "--model-type", type=str, default=config.model.model_type, choices=MODEL_TYPES
    )
    parser.add_argument(
        "--training-type", type=str, default="base", choices=["base", "adversarial", "vehicle"]
    )
    parser.add_argument("--epochs", type=int, default=config.training.epochs)
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
    )
    parser.add_argument("--dataset", type=str, default="fixed", choices=["original", "fixed"])
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        type=str,
        default=config.logging.level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)
    logger.add(
        log_dir / "nids_training_{time}.log",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
    )


def main() -> int:
    args = parse_arguments()
    _configure_logging(args.log_level)

    torch.manual_seed(args.seed)

    logger.info(f"model={args.model_type} training={args.training_type} epochs={args.epochs}")

    dm = NIDSDataModule(dataset=args.dataset)
    dm.setup()
    input_size = len(dm.feature_names)
    logger.info(f"Loaded {input_size} features")

    model = create_model(input_size, args.model_type)

    module_kwargs: dict = {"model": model, "training_type": args.training_type}
    if args.training_type == "adversarial":
        module_kwargs["attack_pattern"] = args.attack_pattern
    module = NIDSLightningModule(**module_kwargs)

    progress_bar = args.log_level in ("DEBUG", "INFO")

    try:
        if config.mlflow.enabled:
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
            experiment_name = args.experiment_name or config.mlflow.experiment_name
            mlflow.set_experiment(experiment_name)
            run_name = f"{args.training_type}_{args.model_type}_{args.epochs}ep"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params(
                    {
                        "model_type": args.model_type,
                        "training_type": args.training_type,
                        "dataset": args.dataset,
                        "epochs": args.epochs,
                        "steps_per_epoch": args.steps_per_epoch,
                        "attack_pattern": args.attack_pattern,
                        "seed": args.seed,
                        "learning_rate": config.training.learning_rate,
                        "batch_size": config.data.batch_size,
                        "pgd_epsilon": config.training.pgd_epsilon,
                        "pgd_steps": config.training.pgd_steps,
                        "pgd_alpha": config.training.pgd_alpha,
                        "lambda_val": config.training.lambda_val,
                    }
                )
                trainer = create_trainer(
                    model_type=args.model_type,
                    training_type=args.training_type,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    experiment_name=experiment_name,
                    mlflow_run_id=run.info.run_id,
                    progress_bar=progress_bar,
                )
                trainer.fit(module, datamodule=dm)
                test_results = trainer.test(module, datamodule=dm)
        else:
            trainer = create_trainer(
                model_type=args.model_type,
                training_type=args.training_type,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                progress_bar=progress_bar,
            )
            trainer.fit(module, datamodule=dm)
            test_results = trainer.test(module, datamodule=dm)

        if test_results:
            logger.info(
                f"Test loss={test_results[0].get('test_loss', 0):.4f} "
                f"acc={test_results[0].get('test_accuracy', 0):.4f}"
            )
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
