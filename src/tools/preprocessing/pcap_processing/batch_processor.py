"""
Batch PCAP Processing Utility with Split Flow Tracking

This module handles splitting large PCAP files and processing them in batches
for scalable feature extraction. Tracks flows that are incomplete due to chunking.
"""

import argparse
import os
import re
import shutil
import subprocess  # nosec
import sys
from collections import defaultdict

import pandas as pd
from loguru import logger
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Split large PCAP files and extract features in batches with split flow tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Process large PCAP with 2GB splits (default)
    python batch_processor.py large_file.pcap output.csv --labels flows.csv

    # Custom split size and window
    python batch_processor.py huge_file.pcap output.csv --labels flows.csv --size-limit 500m --window 15

    # Keep temporary files for debugging
    python batch_processor.py file.pcap output.csv --labels flows.csv --no-cleanup
    """,
    )

    parser.add_argument("input_pcap", help="Path to the input PCAP file")
    parser.add_argument("output_csv", help="Path to save the combined output CSV file")
    parser.add_argument("--labels", required=True, help="Path to the labels CSV file")
    parser.add_argument(
        "--output-dir",
        default="splits",
        help="Directory to store split PCAP files (default: splits)",
    )
    parser.add_argument(
        "--window", type=int, default=10, help="Window size for feature extraction (default: 10)"
    )
    parser.add_argument(
        "--size-limit",
        default="2000m",
        help="Size limit for each split, e.g., '2000m' for 2GB (default: 2000m)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary split files (useful for debugging)",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")

    return parser.parse_args()


class BatchPcapProcessor:
    """
    Handles large PCAP files by splitting them into manageable chunks
    and processing each chunk independently with split flow tracking.
    """

    def __init__(
        self,
        input_pcap: str,
        labels_file: str,
        output_dir: str = "splits",
        window_size: int = 10,
        size_limit: str | int = "2000m",
        verbose: bool = True,
    ):
        """
        Initialize the batch processor.

        Args:
            input_pcap: Path to the large PCAP file
            labels_file: Path to CSV file with labels
            output_dir: Directory for temporary split files
            window_size: Feature extraction window size
            size_limit: Size limit for each split (e.g., '2000m' or 2000 for MB)
            verbose: Enable detailed logging
        """
        self.input_pcap = input_pcap
        self.labels_file = labels_file
        self.output_dir = output_dir
        self.window_size = window_size
        self.size_limit = size_limit
        self.verbose = verbose

        # Track split flows across chunks
        self.split_flow_occurrences: dict[str, int] = defaultdict(int)

    def split_pcap(self) -> list[str]:
        """
        Split the large PCAP file into smaller chunks.

        Returns:
            List of split file paths in chronological order
        """
        # Validate prerequisites
        if not os.path.exists(self.input_pcap):
            raise FileNotFoundError(f"Input PCAP not found: {self.input_pcap}")
        if shutil.which("tcpdump") is None:
            raise RuntimeError("tcpdump not found on PATH; please install tcpdump to split PCAPs")

        os.makedirs(self.output_dir, exist_ok=True)

        # Determine size in MB for tcpdump -C
        size_arg = self.size_limit
        if isinstance(size_arg, str) and size_arg.lower().endswith("m"):
            size_mb = size_arg[:-1]
        else:
            size_mb = str(size_arg)
        if not str(size_mb).isdigit():
            raise ValueError(
                f"Invalid size_limit: {self.size_limit} (expected like '1000m' or '1000')"
            )

        # Use tcpdump to split the file
        split_prefix = os.path.join(self.output_dir, "split_")
        split_command = [
            "tcpdump",
            "-r",
            self.input_pcap,
            "-w",
            split_prefix,
            "-C",
            str(size_mb),
        ]

        try:
            if self.verbose:
                logger.info(f"Splitting {self.input_pcap} into {size_mb}MB chunks...")
            subprocess.run(split_command, check=True, capture_output=not self.verbose)  # nosec
            if self.verbose:
                logger.info(f"PCAP file split into chunks in {self.output_dir}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to split PCAP file: {self.input_pcap}") from e

        # Find all split files
        split_files = self._find_split_files()
        if not split_files:
            raise RuntimeError("No split files were created")

        return split_files

    def _find_split_files(self) -> list[str]:
        """Find and sort split files chronologically."""
        split_files: list[str] = []

        for fname in os.listdir(self.output_dir):
            if fname.startswith("split_") and not fname.endswith(".csv"):
                split_files.append(os.path.join(self.output_dir, fname))

        # Sort files numerically
        def sort_key(filename: str):
            basename = os.path.basename(filename)
            stem, _ext = os.path.splitext(basename)
            m = re.search(r"(\d+)$", stem)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    return float("inf")
            # tcpdump may create base file without suffix first
            if stem == "split_":
                return 0
            # Fallback: try digits at end of full basename
            m2 = re.search(r"(\d+)$", basename)
            if m2:
                try:
                    return int(m2.group(1))
                except ValueError:
                    return float("inf")
            logger.warning(f"Unrecognized split filename ordering: {basename}")
            return float("inf")

        split_files.sort(key=sort_key)
        return split_files

    def process_splits(self, split_files: list[str]) -> pd.DataFrame:
        """
        Process each split file and combine results.

        Args:
            split_files: List of split PCAP file paths in chronological order

        Returns:
            Combined DataFrame with all extracted features from complete flows
        """
        combined_df = pd.DataFrame()
        all_split_flows: list[dict] = []

        pbar = tqdm(
            enumerate(split_files, 1),
            total=len(split_files),
            desc="Processing splits",
            unit="split",
            disable=not self.verbose,
        )

        for i, split_file in pbar:
            try:
                # Run feature extraction on this split
                output_csv, split_report_csv = self._process_single_split(split_file, i)

                if output_csv and os.path.exists(output_csv):
                    # Read and append complete flows
                    df = pd.read_csv(output_csv)
                    combined_df = pd.concat([combined_df, df], ignore_index=True)

                    if self.verbose:
                        pbar.set_postfix({"complete_flows": len(df), "total": len(combined_df)})

                # Track split flows
                if split_report_csv and os.path.exists(split_report_csv):
                    split_df = pd.read_csv(split_report_csv)
                    for _, row in split_df.iterrows():
                        flow_id = row["Flow_ID"]
                        self.split_flow_occurrences[flow_id] += 1
                        all_split_flows.append(dict(row))

            except Exception as e:
                logger.error(f"Failed to process split {i} ({split_file}): {e}")
                continue

        if self.verbose:
            logger.info(
                f"Combined {len(combined_df)} complete flows from {len(split_files)} splits"
            )
            logger.info(f"Tracked {len(self.split_flow_occurrences)} unique split flows")

        # Save split flow analysis
        if all_split_flows:
            self._save_split_flow_analysis(all_split_flows)

        return combined_df

    def _process_single_split(
        self, split_file: str, split_num: int
    ) -> tuple[str | None, str | None]:
        """
        Process a single split file using the feature extractor.

        Returns:
            Tuple of (output_csv_path, split_report_csv_path)
        """
        # Construct module path
        extractor_module = "src.tools.preprocessing.pcap_processing.extractor"

        # Generate output file names
        output_csv = os.path.splitext(split_file)[0] + "_features_with_labels.csv"
        split_report_csv = os.path.splitext(split_file)[0] + "_split_flows.csv"

        # Construct command
        cmd = [
            sys.executable,
            "-m",
            extractor_module,
            split_file,
            "--labels",
            self.labels_file,
            "--window",
            str(self.window_size),
            "--output",
            output_csv,
            "--split-report",
            split_report_csv,
        ]

        if not self.verbose:
            cmd.append("--quiet")

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=not self.verbose,
                text=True,
            )  # nosec

            return (
                output_csv if os.path.exists(output_csv) else None,
                split_report_csv if os.path.exists(split_report_csv) else None,
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Feature extraction failed for split {split_num}: {e}")
            return None, None

    def _save_split_flow_analysis(self, all_split_flows: list[dict]) -> None:
        """Save analysis of split flows across all chunks."""
        analysis_file = os.path.join(self.output_dir, "split_flow_analysis.csv")

        # Create DataFrame from all split flow occurrences
        split_df = pd.DataFrame(all_split_flows)

        # Add occurrence count
        split_df["Split_Count"] = split_df["Flow_ID"].map(self.split_flow_occurrences)

        # Sort by occurrence count (flows split across most chunks first)
        split_df = split_df.sort_values("Split_Count", ascending=False)

        split_df.to_csv(analysis_file, index=False)

        if self.verbose:
            logger.info(f"Split flow analysis saved to {analysis_file}")
            logger.info(f"Most split flow appeared in {split_df['Split_Count'].max()} chunks")

    def cleanup_temp_files(self, split_files: list[str]) -> None:
        """Clean up temporary split files and intermediate CSVs."""
        if self.verbose:
            logger.info("Cleaning up temporary files...")

        files_removed = 0

        # Remove split PCAP files
        for split_file in split_files:
            try:
                if os.path.exists(split_file):
                    os.remove(split_file)
                    files_removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove {split_file}: {e}")

        # Remove intermediate CSV files
        for fname in os.listdir(self.output_dir):
            if fname.endswith("_features_with_labels.csv") or fname.endswith("_split_flows.csv"):
                file_path = os.path.join(self.output_dir, fname)
                try:
                    os.remove(file_path)
                    files_removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")

        if self.verbose:
            logger.info(f"Cleaned up {files_removed} temporary files")

    def process_large_pcap(self, output_csv: str, cleanup: bool = True) -> pd.DataFrame:
        """
        Complete pipeline: split, process, combine, and cleanup.

        Args:
            output_csv: Path for final combined CSV output
            cleanup: Whether to clean up temporary files

        Returns:
            Combined DataFrame with all features from complete flows
        """
        try:
            # Step 1: Split the PCAP file
            split_files = self.split_pcap()

            # Step 2: Process each split
            combined_df = self.process_splits(split_files)

            # Step 3: Save combined results
            if not combined_df.empty:
                combined_df.to_csv(output_csv, index=False)
                if self.verbose:
                    logger.info(f"Combined features saved to {output_csv}")
                    logger.info(f"Final dataset shape: {combined_df.shape}")
            else:
                logger.warning("No features were extracted from any splits")

            # Step 4: Cleanup (optional)
            if cleanup:
                self.cleanup_temp_files(split_files)

            return combined_df

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise


def main():
    """Command-line interface for batch PCAP processing."""

    args = parse_arguments()

    try:
        processor = BatchPcapProcessor(
            input_pcap=args.input_pcap,
            labels_file=args.labels,
            output_dir=args.output_dir,
            window_size=args.window,
            size_limit=args.size_limit,
            verbose=not args.quiet,
        )

        combined_df = processor.process_large_pcap(
            output_csv=args.output_csv, cleanup=not args.no_cleanup
        )

        if not combined_df.empty:
            logger.success("Batch processing completed successfully!")
        else:
            logger.warning("No features were extracted")
            return 1

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
