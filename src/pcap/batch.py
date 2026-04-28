"""Batch PCAP processing for files larger than memory.

Splits with tcpdump, then processes each chunk in-process via FeatureExtractor.
"""

import argparse
import gc
import os
import re
import shutil
import subprocess  # nosec
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.pcap.extractor import FeatureExtractor


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a large PCAP and extract features per chunk"
    )
    parser.add_argument("input_pcap")
    parser.add_argument("output_csv")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--size-limit", default="2000m")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _parse_size_mb(size_limit: str | int) -> str:
    s = str(size_limit).lower()
    if s.endswith("m"):
        s = s[:-1]
    if not s.isdigit():
        raise ValueError(f"Invalid size_limit: {size_limit} (expected '2000m' or '2000')")
    return s


def _split_sort_key(path: str) -> int:
    """Order tcpdump split files by their numeric suffix (split_, split_0, split_1, ...)."""
    stem = Path(path).stem
    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    return 0  # bare 'split_' is the first chunk


class BatchPcapProcessor:
    def __init__(
        self,
        input_pcap: str,
        labels_file: str,
        output_dir: str = "splits",
        window_size: int = 10,
        size_limit: str | int = "2000m",
        verbose: bool = True,
    ):
        self.input_pcap = input_pcap
        self.labels_file = labels_file
        self.output_dir = output_dir
        self.window_size = window_size
        self.size_limit = size_limit
        self.verbose = verbose
        self.split_flow_occurrences: dict[str, int] = defaultdict(int)

    def split_pcap(self) -> list[str]:
        if not os.path.exists(self.input_pcap):
            raise FileNotFoundError(f"Input PCAP not found: {self.input_pcap}")
        if shutil.which("tcpdump") is None:
            raise RuntimeError("tcpdump not found on PATH; install tcpdump to split PCAPs")

        os.makedirs(self.output_dir, exist_ok=True)
        size_mb = _parse_size_mb(self.size_limit)
        split_prefix = os.path.join(self.output_dir, "split_")

        logger.info(f"Splitting {self.input_pcap} into {size_mb}MB chunks…")
        subprocess.run(
            ["tcpdump", "-r", self.input_pcap, "-w", split_prefix, "-C", size_mb],
            check=True,
            capture_output=not self.verbose,
        )  # nosec

        split_files = sorted(
            (
                os.path.join(self.output_dir, f)
                for f in os.listdir(self.output_dir)
                if f.startswith("split_") and not f.endswith(".csv")
            ),
            key=_split_sort_key,
        )
        if not split_files:
            raise RuntimeError("tcpdump produced no split files")
        return split_files

    def process_splits(self, split_files: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        all_split_flows: list[dict] = []

        for split_file in tqdm(
            split_files, desc="Processing splits", unit="split", disable=not self.verbose
        ):
            try:
                df, split_flow_rows = self._process_one(split_file)
                if df is not None and not df.empty:
                    frames.append(df)
                for row in split_flow_rows:
                    self.split_flow_occurrences[row["Flow_ID"]] += 1
                    all_split_flows.append(row)
            except Exception:
                logger.exception(f"Failed processing split {split_file}")

            gc.collect()

        if all_split_flows:
            self._save_split_analysis(all_split_flows)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        logger.info(
            f"Combined {len(combined)} complete flows from {len(split_files)} splits; "
            f"tracked {len(self.split_flow_occurrences)} split flows"
        )
        return combined

    def _process_one(self, split_file: str) -> tuple[pd.DataFrame | None, list[dict]]:
        extractor = FeatureExtractor(
            pcap_file=split_file,
            labels_file=self.labels_file,
            window_size=self.window_size,
            verbose=False,
        )
        extractor.process_packets()
        extractor.match_flows_with_labels()
        df = extractor.compute_features()
        split_rows = extractor.get_split_flow_report().to_dict("records")
        return df, split_rows

    def _save_split_analysis(self, all_split_flows: list[dict]) -> None:
        analysis_file = os.path.join(self.output_dir, "split_flow_analysis.csv")
        split_df = pd.DataFrame(all_split_flows)
        split_df["Split_Count"] = split_df["Flow_ID"].map(self.split_flow_occurrences)
        split_df = split_df.sort_values("Split_Count", ascending=False)
        split_df.to_csv(analysis_file, index=False)
        logger.info(f"Split flow analysis: {analysis_file}")

    def cleanup_temp_files(self, split_files: list[str]) -> None:
        for f in split_files:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

        for fname in os.listdir(self.output_dir):
            if fname.endswith(("_features_with_labels.csv", "_split_flows.csv")):
                try:
                    os.remove(os.path.join(self.output_dir, fname))
                except FileNotFoundError:
                    pass

    def process_large_pcap(self, output_csv: str, cleanup: bool = True) -> pd.DataFrame:
        split_files = self.split_pcap()
        combined_df = self.process_splits(split_files)

        if not combined_df.empty:
            combined_df.to_csv(output_csv, index=False)
            logger.info(f"Combined features → {output_csv} ({combined_df.shape})")
        else:
            logger.warning("No features extracted from any split")

        if cleanup:
            self.cleanup_temp_files(split_files)
        return combined_df


def main() -> int:
    args = _parse_arguments()
    processor = BatchPcapProcessor(
        input_pcap=args.input_pcap,
        labels_file=args.labels,
        output_dir=args.output_dir,
        window_size=args.window,
        size_limit=args.size_limit,
        verbose=not args.quiet,
    )
    combined_df = processor.process_large_pcap(args.output_csv, cleanup=not args.no_cleanup)
    return 0 if not combined_df.empty else 1


if __name__ == "__main__":
    sys.exit(main())
