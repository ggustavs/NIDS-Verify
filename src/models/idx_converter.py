"""
IDX dataset format utilities for Vehicle verification

Convert CSV datasets to IDX format compatible with Vehicle's dataset verification:
https://vehicle-lang.readthedocs.io/en/latest/language/datasets.html
"""
from pathlib import Path
from typing import Tuple

import idx2numpy
import numpy as np
import pandas as pd
from loguru import logger


TIME_INDEX = 0
PROTOCOL_INDEX = 1
DIR_START = 2
DIR_END = 12
FLAGS_START = 12
FLAGS_END = 22
IATS_START = 22
IATS_END = 32
SIZES_START = 32
SIZES_END = 42


def write_idx_file(data: np.ndarray, output_path: str) -> None:
    """
    Write numpy array to IDX file format using idx2numpy library

    Args:
        data: Numpy array to write (any dimension)
        output_path: Path to save IDX file
    """
    # Ensure data is float32 for Vehicle compatibility
    if data.dtype != np.float32:
        logger.info(f"Converting {data.dtype} to float32 for IDX format")
        data = data.astype(np.float32)

    # Write using idx2numpy
    with open(output_path, 'wb') as f:
        idx2numpy.convert_to_file(f, data)

    logger.info(f"✓ Wrote IDX file: {output_path} (shape={data.shape}, dtype={data.dtype})")


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize raw feature values to match Vehicle specification expectations.

    Expected normalization:
    - Protocol: TCP(6) -> 0.0, non-TCP -> 1.0
    - Packet flags: divide by 256
    - Packet sizes: divide by 1000
    - Directions: clamp to [0, 1]
    - Flow duration and IATs: clamp to [0, 1]
    """
    if features.ndim != 2 or features.shape[1] != 42:
        logger.warning(
            f"Unexpected feature shape {features.shape}. Skipping normalization."
        )
        return features.astype(np.float32)

    normalized = features.astype(np.float32, copy=True)

    # Protocol mapping: TCP(6) -> 0, non-TCP -> 1 (idempotent if already 0/1)
    protocol = normalized[:, PROTOCOL_INDEX]
    if np.all((protocol >= 0.0) & (protocol <= 1.0)):
        normalized[:, PROTOCOL_INDEX] = protocol
    else:
        unknown_mask = (protocol != 6.0) & (protocol != 17.0)
        if np.any(unknown_mask):
            logger.warning(f"Unknown protocol values encountered: {np.unique(protocol)}")
        normalized[:, PROTOCOL_INDEX] = np.where(protocol == 6.0, 0.0, 1.0)

    # Directions must be binary (0 = outgoing/forward, 1 = incoming/backward).
    # Accept common encodings (0/1, -1/1, or fractional) and binarize.
    raw_dirs = normalized[:, DIR_START:DIR_END]
    if np.any(raw_dirs < 0.0):
        binarized_dirs = (raw_dirs > 0.0).astype(np.float32)
    else:
        binarized_dirs = (raw_dirs >= 0.5).astype(np.float32)
    normalized[:, DIR_START:DIR_END] = binarized_dirs

    # Normalize flags and sizes
    # Mask to the 6 least-significant TCP flag bits (FIN,SYN,RST,PSH,ACK,URG)
    # to match the specification (ignore ECE/CWR/NS bits that can push values > 255).
    flags_raw = normalized[:, FLAGS_START:FLAGS_END]
    flags_masked = (flags_raw.astype(np.int32) & 0x3F).astype(np.float32)
    normalized[:, FLAGS_START:FLAGS_END] = np.clip(flags_masked / 256.0, 0.0, 1.0)
    normalized[:, SIZES_START:SIZES_END] = np.clip(
        normalized[:, SIZES_START:SIZES_END] / 1000.0, 0.0, 1.0
    )

    # Clamp flow duration and IATs to [0, 1]
    normalized[:, TIME_INDEX] = np.clip(normalized[:, TIME_INDEX], 0.0, 1.0)
    normalized[:, IATS_START:IATS_END] = np.clip(
        normalized[:, IATS_START:IATS_END], 0.0, 1.0
    )

    return normalized


def read_idx_file(file_path: str) -> np.ndarray:
    """
    Read IDX file format into numpy array using idx2numpy library

    Args:
        file_path: Path to IDX file

    Returns:
        Numpy array with data
    """
    with open(file_path, 'rb') as f:
        data = idx2numpy.convert_from_file(f)

    logger.info(f"✓ Read IDX file: {file_path} (shape={data.shape}, dtype={data.dtype})")
    return data


def convert_csv_to_idx(
    csv_path: str,
    output_dir: str,
    features_output: str = "features.idx",
    labels_output: str = "labels.idx",
    max_samples: int | None = None,
) -> Tuple[str, str]:
    """
    Convert CSV dataset to IDX format for Vehicle verification

    Args:
        csv_path: Path to CSV file with features and Label column
        output_dir: Directory to save IDX files
        features_output: Filename for features IDX file
        labels_output: Filename for labels IDX file
        max_samples: Maximum number of samples to convert (None = all)

    Returns:
        Tuple of (features_idx_path, labels_idx_path)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load CSV
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if max_samples is not None:
        df = df.head(max_samples)

    # Extract features and labels
    if 'Label' not in df.columns:
        raise ValueError("CSV must contain 'Label' column")

    # Exclude Label and Flow_ID (string identifier from PCAP extraction)
    exclude_cols = {'Label', 'Flow_ID'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    features = df[feature_cols].values
    labels = df['Label'].values

    # Convert to appropriate dtypes and normalize to match spec
    features = normalize_features(features)

    # Map string labels to binary: BENIGN -> 0, all attacks -> 1
    # Convert to float32 for Vehicle (expects Real type)
    if labels.dtype == object or labels.dtype.kind in ('U', 'S'):
        labels = np.where(labels == 'BENIGN', 0.0, 1.0).astype(np.float32)
    else:
        labels = labels.astype(np.float32)

    labels = np.asarray(labels, dtype=np.float32)

    logger.info(f"Dataset shape: features={features.shape}, labels={labels.shape}")
    logger.info(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
    logger.info(f"Label distribution: {np.bincount(np.asarray(labels, dtype=np.int32))}")

    # Write IDX files
    features_path = str(Path(output_dir) / features_output)
    labels_path = str(Path(output_dir) / labels_output)

    write_idx_file(features, features_path)
    write_idx_file(labels, labels_path)

    return features_path, labels_path


def normalize_idx_dataset(
    features_path: str,
    labels_path: str,
) -> Tuple[str, str]:
    """
    Normalize an existing IDX dataset in-place to match the specification.

    Args:
        features_path: Path to features IDX file
        labels_path: Path to labels IDX file

    Returns:
        Tuple of (features_idx_path, labels_idx_path)
    """
    features = read_idx_file(features_path)
    labels = read_idx_file(labels_path)

    features = normalize_features(features)
    labels = labels.astype(np.float32, copy=False)

    write_idx_file(features, features_path)
    write_idx_file(labels, labels_path)

    return features_path, labels_path


def prepare_vehicle_dataset(
    csv_path: str,
    output_dir: str = "data/idx",
    subset_name: str = "test",
    max_samples: int | None = None,
) -> Tuple[str, str]:
    """
    Prepare dataset in IDX format for Vehicle verification

    Args:
        csv_path: Path to CSV dataset
        output_dir: Directory for IDX files
        subset_name: Name for this subset (e.g., 'test', 'train')
        max_samples: Limit number of samples

    Returns:
        Tuple of (features_idx_path, labels_idx_path)
    """
    features_file = f"{subset_name}_features.idx"
    labels_file = f"{subset_name}_labels.idx"

    return convert_csv_to_idx(
        csv_path,
        output_dir,
        features_file,
        labels_file,
        max_samples
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python idx_converter.py <csv_path> <output_dir> [--max-samples N]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2]
    max_samples = None

    # Parse optional --max-samples argument
    if len(sys.argv) >= 5 and sys.argv[3] == "--max-samples":
        max_samples = int(sys.argv[4])

    features_path, labels_path = prepare_vehicle_dataset(
        csv_path, output_dir, max_samples=max_samples
    )
    logger.info(f"✓ IDX files created: {features_path}, {labels_path}")
