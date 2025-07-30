"""
Batch PCAP Processing Utility

This module handles splitting large PCAP files and processing them in batches
for scalable feature extraction. Includes automated cleanup and result aggregation.
"""

import argparse
import logging
import os
import subprocess  # nosec

import pandas as pd

logger = logging.getLogger(__name__)


class BatchPcapProcessor:
    """
    Handles large PCAP files by splitting them into manageable chunks
    and processing each chunk independently.
    """

    def __init__(
        self,
        input_pcap: str,
        labels_file: str,
        output_dir: str = "splits",
        window_size: int = 10,
        size_limit: str = "1000m",
        verbose: bool = True,
    ):
        """
        Initialize the batch processor.

        Args:
            input_pcap: Path to the large PCAP file
            labels_file: Path to CSV file with labels
            output_dir: Directory for temporary split files
            window_size: Feature extraction window size
            size_limit: Size limit for each split (e.g., '1000m' for 1GB)
            verbose: Enable detailed logging
        """
        self.input_pcap = input_pcap
        self.labels_file = labels_file
        self.output_dir = output_dir
        self.window_size = window_size
        self.size_limit = size_limit
        self.verbose = verbose
        self.temp_dir_created = False

    def split_pcap(self) -> list[str]:
        """
        Split the large PCAP file into smaller chunks.

        Returns:
            List of split file paths
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Use tcpdump to split the file
        split_prefix = os.path.join(self.output_dir, "split_")
        split_command = [
            "tcpdump",
            "-r",
            self.input_pcap,
            "-w",
            split_prefix,
            "-C",
            self.size_limit[:-1],  # Remove 'm' suffix
        ]

        try:
            if self.verbose:
                logger.info(f"Splitting {self.input_pcap} into {self.size_limit} chunks...")
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
        """Find and sort split files."""
        split_files = []

        for file in os.listdir(self.output_dir):
            if file.startswith("split_") and not file.endswith(".csv"):
                split_files.append(os.path.join(self.output_dir, file))

        # Sort files numerically if possible
        def sort_key(filename):
            basename = os.path.basename(filename)
            if basename == "split_":
                return 0
            try:
                # Extract number from filename like "split_01", "split_02", etc.
                parts = basename.split("_")
                if len(parts) > 1 and parts[-1].isdigit():
                    return int(parts[-1])
            except (ValueError, IndexError, AttributeError):
                logging.logger.warning(f"Failed to parse split filename: {basename}")
                pass
            return float("inf")

        split_files.sort(key=sort_key)
        return split_files

    def process_splits(self, split_files: list[str]) -> pd.DataFrame:
        """
        Process each split file and combine results.

        Args:
            split_files: List of split PCAP file paths

        Returns:
            Combined DataFrame with all extracted features
        """
        combined_df = pd.DataFrame()
        processed_files = []

        for i, split_file in enumerate(split_files, 1):
            if self.verbose:
                logger.info(f"Processing split {i}/{len(split_files)}: {split_file}")

            try:
                # Run feature extraction on this split
                output_csv = self._process_single_split(split_file)

                if output_csv and os.path.exists(output_csv):
                    # Read and append results
                    df = pd.read_csv(output_csv)
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                    processed_files.append(output_csv)

                    if self.verbose:
                        logger.info(f"Split {i} processed: {len(df)} flows extracted")
                else:
                    logger.warning(f"No output generated for split {i}: {split_file}")

            except Exception as e:
                logger.error(f"Failed to process split {i} ({split_file}): {e}")
                continue

        if self.verbose:
            logger.info(
                f"Combined {len(combined_df)} total flows from {len(processed_files)} splits"
            )

        return combined_df

    def _process_single_split(self, split_file: str) -> str | None:
        """Process a single split file using the feature extractor."""
        # Construct command to run feature extraction
        cmd = [
            "python",
            "-m",
            "tools.pcap_processing.extractor",
            split_file,
            "--labels",
            self.labels_file,
            "--window",
            str(self.window_size),
        ]

        if not self.verbose:
            cmd.append("--quiet")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # nosec

            # Expected output CSV file
            output_csv = os.path.splitext(split_file)[0] + "_features_with_labels.csv"

            if os.path.exists(output_csv):
                return output_csv
            else:
                logger.warning(f"Expected output file not found: {output_csv}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Feature extraction failed for {split_file}: {e}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            return None

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
        for file in os.listdir(self.output_dir):
            if file.endswith("_features_with_labels.csv"):
                try:
                    file_path = os.path.join(self.output_dir, file)
                    os.remove(file_path)
                    files_removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")

        # Remove output directory if empty and it's the default
        if self.output_dir == "splits" and os.path.exists(self.output_dir):
            try:
                if not os.listdir(self.output_dir):
                    os.rmdir(self.output_dir)
                    if self.verbose:
                        logger.info("Removed empty splits directory")
            except Exception as e:
                logger.warning(f"Failed to remove directory {self.output_dir}: {e}")

        if self.verbose:
            logger.info(f"Cleaned up {files_removed} temporary files")

    def process_large_pcap(self, output_csv: str, cleanup: bool = True) -> pd.DataFrame:
        """
        Complete pipeline: split, process, combine, and cleanup.

        Args:
            output_csv: Path for final combined CSV output
            cleanup: Whether to clean up temporary files

        Returns:
            Combined DataFrame with all features
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
    parser = argparse.ArgumentParser(
        description="Split large PCAP files and extract features in batches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process large PCAP with 1GB splits
  python batch_processor.py large_file.pcap output.csv --labels flows.csv

  # Custom split size and window
  python batch_processor.py huge_file.pcap output.csv --labels flows.csv --size-limit 2000m --window 15

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
        default="1000m",
        help="Size limit for each split, e.g., '1000m' for 1GB (default: 1000m)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary split files (useful for debugging)",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

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
            print("✅ Batch processing completed successfully!")
            print(f"📁 Combined output saved to: {args.output_csv}")
            print(f"📊 Total flows extracted: {len(combined_df)}")
        else:
            print("⚠️ No features were extracted")
            return 1

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
