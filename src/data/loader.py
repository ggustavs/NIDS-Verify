"""Unified data loading for NIDS datasets (PyTorch)."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

from src.config import config


class DatasetType(Enum):
    """Available datasets for NIDS training."""

    ORIGINAL = "original"
    FIXED = "fixed"

    @property
    def description(self) -> str:
        descriptions = {
            "original": "Original DoS dataset (preprocessed CSVs)",
            "fixed": "Extracted PCAP features from fixed data",
        }
        return descriptions.get(self.value, "Unknown dataset")


@dataclass
class _DatasetConfig:
    name: str
    train_path: Path
    test_path: Path

    def validate(self) -> None:
        if not self.train_path.exists():
            raise FileNotFoundError(f"Train file not found: {self.train_path}")
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test file not found: {self.test_path}")


def _get_dataset_config(dataset_type: str, data_dir: str) -> _DatasetConfig:
    data_path = Path(data_dir)
    configs = {
        "original": _DatasetConfig(
            name="Original DoS",
            train_path=data_path / "preprocessed-dos-train.csv",
            test_path=data_path / "preprocessed-dos-test.csv",
        ),
        "fixed": _DatasetConfig(
            name="Fixed Flows",
            train_path=data_path / "wednesday_train_binary.csv",
            test_path=data_path / "wednesday_test_binary.csv",
        ),
    }

    if dataset_type not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown dataset '{dataset_type}'. Available: {available}")

    config_obj = configs[dataset_type]
    config_obj.validate()
    return config_obj


class _NDArrayDataset(Dataset):
    """Simple Dataset wrapping numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    @property
    def mean(self):
        return torch.as_tensor(np.mean(self.X, axis=0))

    @property
    def std(self):
        return torch.as_tensor(np.std(self.X, axis=0))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y_arr = np.asarray(self.y[idx], dtype=np.int64)
        return torch.from_numpy(self.X[idx]), torch.as_tensor(y_arr, dtype=torch.long)


class DataLoader:
    """Unified data loader for NIDS datasets."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or config.data.data_dir
        self.scaler = MinMaxScaler()
        self.feature_names: List[str] = []

    def load_data(
        self,
        dataset: str = "fixed",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, List[str]]:
        logger.info(f"Loading dataset: {dataset}")

        dataset_config = _get_dataset_config(dataset, self.data_dir)
        logger.info(f"Using dataset: {dataset_config.name}")

        train_df = pd.read_csv(dataset_config.train_path)
        test_df = pd.read_csv(dataset_config.test_path)
        logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

        self._binarize_direction_columns(train_df)
        self._binarize_direction_columns(test_df)

        label_col = train_df.columns[-1]
        self.feature_names = [c for c in train_df.columns if c != label_col]
        logger.info(f"Features: {len(self.feature_names)} columns")

        # Extract features and labels, ensuring numeric types
        X_train = train_df[self.feature_names].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
        y_train = train_df[label_col].to_numpy()
        X_test = test_df[self.feature_names].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
        y_test = test_df[label_col].to_numpy()

        # Handle infinite values - replace with 0
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=test_size,
            random_state=random_state,
            stratify=y_train,
        )

        if (X_train.min() < 0).any() or (X_train.max() > 1).any():
            logger.warning("Feature values not in [0, 1] - applying MinMaxScaler")
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        logger.info(
            f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
        )

        train_loader = self._create_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_loader(X_val, y_val, shuffle=False)
        test_loader = self._create_loader(X_test, y_test, shuffle=False)

        return train_loader, val_loader, test_loader, self.feature_names

    @staticmethod
    def _binarize_direction_columns(df: pd.DataFrame) -> None:
        dir_cols = [col for col in df.columns if col.startswith("Pkt_Direction")]
        if not dir_cols:
            return

        dir_values = df[dir_cols].to_numpy(dtype=np.float32, copy=True)
        if np.any(dir_values < 0.0):
            binarized = (dir_values > 0.0).astype(np.float32)
        else:
            binarized = (dir_values >= 0.5).astype(np.float32)

        df.loc[:, dir_cols] = binarized

    def _create_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = False,
    ) -> TorchDataLoader:
        dataset = _NDArrayDataset(X, y)
        return TorchDataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
