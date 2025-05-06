import os
import subprocess
import pandas as pd
import argparse

def split_pcap(input_pcap, output_dir, size_limit="1000m"):
    """Split the PCAP file into smaller files of specified size."""
    os.makedirs(output_dir, exist_ok=True)
    split_command = [
        "tcpdump", "-r", input_pcap, "-w", os.path.join(output_dir, "split_"), "-C", size_limit[:-1]
    ]
    subprocess.run(split_command, check=True)
    print(f"PCAP file split into chunks in {output_dir}")

def process_splits(output_dir, labels_file, window_size, verbose):
    """Run extract_features.py on all split files and collect the output."""
    split_files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("split_") and not f.endswith("_features_with_labels.csv")],
        key=lambda x: (0 if x == "split_" else int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else float("inf"))
    )
    combined_df = pd.DataFrame()

    for split_file in split_files:
        split_path = os.path.join(output_dir, split_file)
        print(f"Processing {split_path}...")
        output_csv = split_path + "_features_with_labels.csv"
        uv_command = [
            "uv", "run", "tools/extract_features.py", split_path,
            "--labels", labels_file, "--window", str(window_size)
        ]
        if not verbose:
            uv_command.append("--not-verbose")
        subprocess.run(uv_command, check=True)

        # Append the resulting CSV to the combined dataframe
        if os.path.exists(output_csv):
            try:
                df = pd.read_csv(output_csv)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                print(f"Warning: Failed to read {output_csv}. Error: {e}")
        else:
            print(f"Warning: Output CSV {output_csv} not found.")

    # Cleanup: Remove split files and temporary CSV files
    for split_file in split_files:
        split_path = os.path.join(output_dir, split_file)
        output_csv = split_path + "_features_with_labels.csv"
        if os.path.exists(split_path):
            os.remove(split_path)
        if os.path.exists(output_csv):
            os.remove(output_csv)

    # Remove the output directory if it is the default and empty
    if output_dir == "splits" and not os.listdir(output_dir):
        os.rmdir(output_dir)

    return combined_df

def main():
    parser = argparse.ArgumentParser(description="Split PCAP file and extract features.")
    parser.add_argument("input_pcap", help="Path to the input PCAP file")
    parser.add_argument("output_csv", help="Path to save the combined output CSV file")
    parser.add_argument("--labels", required=True, help="Path to the labels CSV file")
    parser.add_argument("--output_dir", default="splits", help="Directory to store split PCAP files")
    parser.add_argument("--window", type=int, default=10, help="Window size for feature extraction")
    parser.add_argument("--size_limit", default="1000m", help="Size limit for each split (e.g., '1000m' for 1GB)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Step 1: Split the PCAP file
    split_pcap(args.input_pcap, args.output_dir, args.size_limit)

    # Step 2: Process each split and combine results
    combined_df = process_splits(args.output_dir, args.labels, args.window, args.verbose)

    # Step 3: Save the combined dataframe to the output CSV
    combined_df.to_csv(args.output_csv, index=False)
    print(f"Combined features saved to {args.output_csv}")

if __name__ == "__main__":
    main()
