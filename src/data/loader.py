"""
Data loading and preprocessing utilities
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config import config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Data loader for NIDS datasets"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or config.data.data_dir
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_data(self, dataset: str = "fixed", test_size: float = 0.2, random_state: int = 42) -> Tuple[
        tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]
    ]:
        """
        Args:
            dataset: Dataset to use ("original" or "fixed")
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset, feature_names)
        """
        logger.info(f"Loading DoS dataset: {dataset}")

        if dataset == "original":
            return self._load_original_data(test_size, random_state)
        elif dataset == "fixed":
            return self._load_fixed_data(test_size, random_state)
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Choose 'original' or 'fixed'")

    def _load_original_data(self, test_size: float, random_state: int) -> Tuple[
        tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]
    ]:
        """Load the original researcher's original data (already split)"""
        logger.info("Loading original DoS dataset...")

        # Load original data
        train_path = os.path.join(self.data_dir, "preprocessed-dos-train.csv")
        test_path = os.path.join(self.data_dir, "preprocessed-dos-test.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"original data not found. Please run preprocessing first.")

        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info(f"Loaded train data: {train_df.shape}, test data: {test_df.shape}")

        # Separate features and labels (original data uses numeric column names)
        # Assume last column is label
        label_col = train_df.columns[-1]
        self.feature_names = [col for col in train_df.columns if col != label_col]

        X_train = train_df[self.feature_names].values
        y_train = train_df[label_col].values
        X_test = test_df[self.feature_names].values
        y_test = test_df[label_col].values

        # Split training data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=test_size,
            random_state=random_state, stratify=y_train
        )

        if (X_train.min() < 0).any() or (X_train.max() > 1).any():
            logger.warning("Feature values are not normalized to [0, 1] range. Normalizing now.")
            # Normalize features
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"Feature names: {len(self.feature_names)} features")

        # Create TensorFlow datasets
        train_dataset = self._create_dataset(X_train, y_train, shuffle=True)
        val_dataset = self._create_dataset(X_val, y_val, shuffle=False)
        test_dataset = self._create_dataset(X_test, y_test, shuffle=False)

        return train_dataset, val_dataset, test_dataset, self.feature_names

    def _load_fixed_data(self, test_size: float, random_state: int) -> Tuple[
        tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]
    ]:
        """Load the CIC Wednesday dataset (already split into train/test files)"""
        logger.info("Loading CIC Wednesday DoS dataset...")

        # Load pre-split CIC Wednesday data
        train_path = os.path.join(self.data_dir, "CICWednesdayData", "pos_neg", "cic_ids_2017_train.csv")
        test_path = os.path.join(self.data_dir, "CICWednesdayData", "pos_neg", "cic_ids_2017_test.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"CIC Wednesday train data not found at {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"CIC Wednesday test data not found at {test_path}")

        # Load data
        logger.info("Loading pre-split train and test datasets...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info(f"Loaded train data: {train_df.shape}, test data: {test_df.shape}")

        # Separate features and labels
        if 'Label' in train_df.columns:
            label_col = 'Label'
        elif 'label' in train_df.columns:
            label_col = 'label'
        else:
            # Assume last column is label
            label_col = train_df.columns[-1]

        self.feature_names = [col for col in train_df.columns if col != label_col]

        X_train = train_df[self.feature_names].values
        y_train = train_df[label_col].values
        X_test = test_df[self.feature_names].values
        y_test = test_df[label_col].values

        # Split training data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=test_size,
            random_state=random_state, stratify=y_train
        )

        if (X_train.min() < 0).any() or (X_train.max() > 1).any():
            logger.warning("Feature values are not normalized to [0, 1] range. Normalizing now.")
            # Normalize features
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"Feature names: {len(self.feature_names)} features")

        # Create TensorFlow datasets
        train_dataset = self._create_dataset(X_train, y_train, shuffle=True)
        val_dataset = self._create_dataset(X_val, y_val, shuffle=False)
        test_dataset = self._create_dataset(X_test, y_test, shuffle=False)

        return train_dataset, val_dataset, test_dataset, self.feature_names

    def _create_dataset(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset from numpy arrays"""
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.cast(X, tf.float32),
            tf.cast(y, tf.int32)
        ))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=config.data.shuffle_buffer_size)

        dataset = dataset.batch(config.data.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_input_size(self) -> int:
        """Get the number of input features"""
        return len(self.feature_names)
