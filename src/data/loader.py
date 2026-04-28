"""Lightning DataModule for NIDS datasets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from src.config import config

_LABEL_COL = "Label"


@dataclass
class _DatasetSpec:
    name: str
    train_path: Path
    test_path: Path


_DATASETS: dict[str, callable] = {
    "original": lambda d: _DatasetSpec(
        "Original DoS",
        Path(d) / "preprocessed-dos-train.csv",
        Path(d) / "preprocessed-dos-test.csv",
    ),
    "fixed": lambda d: _DatasetSpec(
        "Fixed Flows",
        Path(d) / "wednesday_train_binary.csv",
        Path(d) / "wednesday_test_binary.csv",
    ),
}


class _TensorDataset(Dataset):
    """In-memory dataset over preconverted tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class NIDSDataModule(pl.LightningDataModule):
    """Loads CIC-IDS CSVs, normalises to [0,1], yields train/val/test DataLoaders."""

    def __init__(
        self,
        dataset: str = "fixed",
        data_dir: str | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__()
        if dataset not in _DATASETS:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(_DATASETS)}")
        self.dataset = dataset
        self.data_dir = data_dir or config.data.data_dir
        self.batch_size = batch_size or config.data.batch_size
        self.num_workers = num_workers if num_workers is not None else config.data.num_workers
        self.val_size = val_size
        self.random_state = random_state

        self.feature_names: list[str] = []
        self._train_ds: _TensorDataset | None = None
        self._val_ds: _TensorDataset | None = None
        self._test_ds: _TensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        spec = _DATASETS[self.dataset](self.data_dir)
        if not spec.train_path.exists():
            raise FileNotFoundError(spec.train_path)
        if not spec.test_path.exists():
            raise FileNotFoundError(spec.test_path)

        logger.info(f"Loading dataset: {spec.name}")
        train_df = pd.read_csv(spec.train_path)
        test_df = pd.read_csv(spec.test_path)
        logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

        if _LABEL_COL not in train_df.columns:
            raise ValueError(f"Train CSV missing '{_LABEL_COL}' column")
        if _LABEL_COL not in test_df.columns:
            raise ValueError(f"Test CSV missing '{_LABEL_COL}' column")

        self.feature_names = [c for c in train_df.columns if c != _LABEL_COL]

        X_train = self._extract_features(train_df)
        y_train = train_df[_LABEL_COL].to_numpy()
        X_test = self._extract_features(test_df)
        y_test = test_df[_LABEL_COL].to_numpy()

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_train,
        )

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        self._train_ds = _TensorDataset(X_train, y_train)
        self._val_ds = _TensorDataset(X_val, y_val)
        self._test_ds = _TensorDataset(X_test, y_test)

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test_ds, shuffle=False)

    def _make_loader(self, ds: _TensorDataset | None, shuffle: bool) -> DataLoader:
        if ds is None:
            raise RuntimeError("setup() must be called before requesting dataloaders")
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
