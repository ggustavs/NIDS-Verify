"""
Data loading and preprocessing utilities (PyTorch)
"""
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

from src.config import config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class _NDArrayDataset(Dataset):
    """Simple Dataset wrapping numpy arrays"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:  # noqa: D401
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        # Ensure y is converted correctly whether scalar or array
        y_arr = np.asarray(y, dtype=np.int64)
        return torch.from_numpy(x), torch.as_tensor(y_arr, dtype=torch.long)


class DataLoader:
    """Data loader for NIDS datasets (returns PyTorch DataLoaders)"""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or config.data.data_dir
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def load_data(
        self, dataset: str = "fixed", test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, List[str]]:
        """Load datasets and return PyTorch DataLoaders (train/val/test)."""
        logger.info(f"Loading DoS dataset: {dataset}")

        if dataset == "fixed":
            return self._load_fixed_data(test_size, random_state)
        if dataset == "original":
            return self._load_original_data(test_size, random_state)
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'original' or 'fixed'")

    def _load_original_data(
        self, test_size: float, random_state: int
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, List[str]]:
        logger.info("Loading original DoS dataset...")

        train_path = os.path.join(self.data_dir, "preprocessed-dos-train.csv")
        test_path = os.path.join(self.data_dir, "preprocessed-dos-test.csv")
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("original data not found. Please run preprocessing first.")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded train data: {train_df.shape}, test data: {test_df.shape}")

        label_col = train_df.columns[-1]
        self.feature_names = [c for c in train_df.columns if c != label_col]

        X_train = train_df[self.feature_names].to_numpy()
        y_train = train_df[label_col].to_numpy()
        X_test = test_df[self.feature_names].to_numpy()
        y_test = test_df[label_col].to_numpy()

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=random_state, stratify=y_train
        )

        if (X_train.min() < 0).any() or (X_train.max() > 1).any():
            logger.warning("Feature values not in [0, 1]. Applying StandardScaler normalization.")
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        logger.info(
            f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
        )
        logger.info(f"Feature names: {len(self.feature_names)} features")

        train_loader = self._create_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_loader(X_val, y_val, shuffle=False)
        test_loader = self._create_loader(X_test, y_test, shuffle=False)

        return train_loader, val_loader, test_loader, self.feature_names

    def _load_fixed_data(
        self, test_size: float, random_state: int
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, List[str]]:
        logger.info("Loading CIC Wednesday DoS dataset...")

        train_path = os.path.join(
            self.data_dir, "CICWednesdayData", "pos_neg", "cic_ids_2017_train.csv"
        )
        test_path = os.path.join(
            self.data_dir, "CICWednesdayData", "pos_neg", "cic_ids_2017_test.csv"
        )

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"CIC Wednesday train data not found at {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"CIC Wednesday test data not found at {test_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded train data: {train_df.shape}, test data: {test_df.shape}")

        if "Label" in train_df.columns:
            label_col = "Label"
        elif "label" in train_df.columns:
            label_col = "label"
        else:
            label_col = train_df.columns[-1]

        self.feature_names = [c for c in train_df.columns if c != label_col]

        X_train = train_df[self.feature_names].to_numpy()
        y_train = train_df[label_col].to_numpy()
        X_test = test_df[self.feature_names].to_numpy()
        y_test = test_df[label_col].to_numpy()

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=random_state, stratify=y_train
        )

        if (X_train.min() < 0).any() or (X_train.max() > 1).any():
            logger.warning("Feature values not in [0, 1]. Applying StandardScaler normalization.")
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        logger.info(
            f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
        )
        logger.info(f"Feature names: {len(self.feature_names)} features")

        train_loader = self._create_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_loader(X_val, y_val, shuffle=False)
        test_loader = self._create_loader(X_test, y_test, shuffle=False)

        return train_loader, val_loader, test_loader, self.feature_names

    def _create_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> TorchDataLoader:
        ds = _NDArrayDataset(X, y)
        return TorchDataLoader(
            ds,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=torch.cuda.is_available(),
        )

    def get_input_size(self) -> int:
        return len(self.feature_names)
