"""
Feature Extraction Tool with Timestamp-Based Flow Matching

This module extracts machine learning features from PCAP files using precise
timestamp and duration information from CSV labels to accurately recreate flows.

Key Features:
- Timestamp-based flow matching with microsecond precision
- Flow completeness detection for chunked PCAP processing
- Time-window based bidirectional packet matching
- Comprehensive feature extraction for ML training
- Memory-efficient batch processing

Workflow:
1. Index all packets by their 5-tuple
2. Read and sort flow labels by timestamp
3. For each flow, match packets within its time window (timestamp to timestamp+duration)
4. Determine packet direction relative to flow initiator (from Flow ID)
5. Identify complete vs split flows (incomplete due to chunking)
6. Extract features only from complete flows
"""

import argparse
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.utils import RawPcapReader
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments for feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract ML features from PCAP files with timestamp-based flow matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Basic feature extraction
        python extractor.py sample.pcap --labels flows.csv --window 10

        # Generate split flow report
        python extractor.py sample.pcap --labels flows.csv --split-report splits.csv

        # Quiet mode
        python extractor.py sample.pcap --labels flows.csv --quiet
        """,
    )
    parser.add_argument("pcap_file", help="Path to the PCAP file")
    parser.add_argument("--labels", required=True, help="Path to the labels CSV file")
    parser.add_argument(
        "--window", type=int, default=10, help="Window size for feature extraction (default: 10)"
    )
    parser.add_argument("--output", help="Output CSV file path (auto-generated if not specified)")
    parser.add_argument("--split-report", help="Generate report of split flows to this file")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")

    return parser.parse_args()


@dataclass
class FlowInfo:
    """Information about a flow from the CSV labels."""

    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    timestamp: float  # Start time in Unix seconds
    duration: float  # Duration in seconds
    fwd_packets: int
    bwd_packets: int
    total_packets: int
    label: str
    is_complete: bool = True  # Whether all packets were captured


def parse_flow_id(flow_id: str) -> tuple[str, str, int, int, int]:
    """Parse Flow ID into 5-tuple components."""
    parts = flow_id.split("-")
    if len(parts) != 5:
        raise ValueError(f"Invalid Flow ID format: {flow_id}")
    src_ip, dst_ip = parts[0], parts[1]
    src_port, dst_port, protocol = int(parts[2]), int(parts[3]), int(parts[4])
    return src_ip, dst_ip, src_port, dst_port, protocol


def normalize_flow_duration(raw_value: float | None) -> float:
    """Normalize flow duration from CSV to seconds (CIC CSV stores microseconds)."""
    if raw_value is None or pd.isna(raw_value):
        return 0.0
    try:
        val = float(raw_value)
        # CIC CSV durations are in microseconds
        return val / 1_000_000.0
    except (TypeError, ValueError):
        return 0.0


class FeatureExtractor:
    """
    Timestamp-based feature extractor for NIDS packet data.

    Handles PCAP file processing with precise flow matching using CSV labels.
    """

    def __init__(
        self,
        pcap_file: str,
        labels_file: str,
        window_size: int = 10,
        verbose: bool = True,
    ):
        """
        Initialize the feature extractor.

        Args:
            pcap_file: Path to the PCAP file to process
            labels_file: Path to CSV file with flow labels (required)
            window_size: Number of packets to include in feature window
            verbose: Enable detailed progress output
        """
        self.pcap_file = pcap_file
        self.labels_file = labels_file
        self.window_size = window_size
        self.verbose = verbose

        # Packet storage: indexed by 5-tuple for fast lookup
        # Stores packets in both possible directions of the 5-tuple
        self.packets_by_tuple: dict[tuple, list[tuple]] = defaultdict(list)

        # Flow storage: Complete flows ready for feature extraction
        self.complete_flows: dict[str, tuple[FlowInfo, list[tuple]]] = {}
        self.split_flows: dict[str, FlowInfo] = {}  # Incomplete flows

        if verbose:
            logger.info(f"Initialized FeatureExtractor for {pcap_file}")
            logger.info(f"Labels: {labels_file}, Window size: {window_size}")

    def process_packets(self) -> None:
        """
        Index all packets by their 5-tuple using fast RawPcapReader.

        Direction is determined later when matching against flows using time windows.
        Uses batch processing with RawPcapReader for optimal memory/speed balance.
        """
        if self.verbose:
            logger.info(f"Processing PCAP: {self.pcap_file}")

        packet_count = 0
        # Use RawPcapReader for faster processing - only parse packets we need
        try:
            for raw_data, metadata in tqdm(
                RawPcapReader(self.pcap_file),
                desc="Indexing packets",
                unit="pkt",
                disable=not self.verbose,
            ):
                # Parse packet from raw data
                try:
                    pkt = Ether(raw_data)
                except Exception:  # nosec B112
                    continue  # Skip malformed packets

                if IP not in pkt:
                    continue

                packet_data = self._extract_packet_info(pkt, metadata)
                if packet_data is None:
                    continue

                five_tuple, packet_record = packet_data
                self.packets_by_tuple[five_tuple].append(packet_record)
                packet_count += 1

        except Exception as e:
            raise RuntimeError(f"Failed to process PCAP file {self.pcap_file}") from e

        # Sort packets by timestamp for each 5-tuple
        if self.verbose:
            logger.info("Sorting packets by timestamp...")
        for five_tuple in self.packets_by_tuple:
            self.packets_by_tuple[five_tuple].sort(key=lambda pkt: pkt[0])

        if self.verbose:
            logger.info(f"Indexed {packet_count:,} packets")
            logger.info(f"Unique 5-tuples: {len(self.packets_by_tuple)}")

    def _extract_packet_info(self, pkt, metadata=None) -> tuple[tuple, tuple] | None:
        """
        Extract packet information and organize by 5-tuple.

        Args:
            pkt: Parsed packet
            metadata: Tuple of (sec, usec, caplen, length) from RawPcapReader

        Returns:
            Tuple of (5-tuple, packet_record) where packet_record is
            (timestamp, protocol, size, flags), or None if packet is invalid.
        """
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        # Get timestamp from metadata if available (RawPcapReader), otherwise from packet
        if metadata is not None:
            # RawPcapReader returns variable-length metadata tuples depending on PCAP/PCAPNG.
            sec = metadata[0]
            usec = metadata[1]
            timestamp = float(sec) + float(usec) / 1_000_000.0
        else:
            timestamp = float(pkt.time) if pkt.time else 0.0

        protocol = pkt[IP].proto
        # Use IP packet length (excludes Ethernet header) to match spec sizes (e.g., 40/52 bytes).
        pkt_size = len(pkt[IP])

        # Extract port and flags information
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            # Mask to 6-bit TCP flags (FIN,SYN,RST,PSH,ACK,URG) to avoid ECN/NS bits.
            pkt_flags = int(pkt[TCP].flags) & 0x3F
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
            pkt_flags = 0
        else:
            return None

        if src_port is None or dst_port is None:
            return None

        five_tuple = (src_ip, dst_ip, src_port, dst_port, protocol)
        packet_record = (timestamp, protocol, pkt_size, pkt_flags)

        return five_tuple, packet_record

    def match_flows_with_labels(self) -> None:
        """
        Match packets to flows using precise timestamps and durations from CSV.

        Identifies complete vs split flows based on packet counts and time windows.
        """
        # Load and sort labels by timestamp
        try:
            labels_df = pd.read_csv(self.labels_file)
            labels_df.columns = labels_df.columns.str.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read labels file {self.labels_file}") from e

        # Parse timestamps
        labels_df["Timestamp_Unix"] = (
            pd.to_datetime(labels_df["Timestamp"], format="%Y-%m-%d %H:%M:%S.%f").astype("int64")
            / 1e9
        )

        labels_df = labels_df.sort_values("Timestamp_Unix")

        if self.verbose:
            logger.info(f"Loaded {len(labels_df)} flow labels")
            logger.info(
                f"Time range: {labels_df['Timestamp'].iloc[0]} to {labels_df['Timestamp'].iloc[-1]}"
            )

        # Get PCAP time range
        pcap_min, pcap_max = self._get_pcap_time_range()
        if self.verbose and pcap_min is not None and pcap_max is not None:
            logger.info(
                f"PCAP time range: {pd.to_datetime(pcap_min, unit='s')} to "
                f"{pd.to_datetime(pcap_max, unit='s')}"
            )

        # Match each flow from labels using iterrows (simpler than managing namedtuple field name mapping)
        for label_idx, row in tqdm(
            labels_df.iterrows(),
            total=len(labels_df),
            desc="Matching flows",
            unit="flow",
            disable=not self.verbose,
        ):
            flow_info = FlowInfo(
                flow_id=row["Flow ID"],
                src_ip=row["Src IP"],
                dst_ip=row["Dst IP"],
                src_port=int(row["Src Port"]),
                dst_port=int(row["Dst Port"]),
                protocol=int(row["Protocol"]),
                timestamp=float(row["Timestamp_Unix"]),
                duration=normalize_flow_duration(row["Flow Duration"]),
                fwd_packets=int(row["Total Fwd Packet"]),
                bwd_packets=int(row["Total Bwd packets"]),
                total_packets=int(row["Total Fwd Packet"]) + int(row["Total Bwd packets"]),
                label=row["Label"],
            )

            # Embed the label_idx for uniqueness when multiple flows have same 5-tuple
            flow_info.flow_id = f"{label_idx}:{flow_info.flow_id}"

            matched_packets = self._match_flow_packets(flow_info, pcap_min, pcap_max)

            if matched_packets is not None:
                if flow_info.is_complete:
                    # Use unique key: label_idx to avoid overwriting flows with same 5-tuple
                    self.complete_flows[str(label_idx)] = (flow_info, matched_packets)
                else:
                    self.split_flows[str(label_idx)] = flow_info

        if self.verbose:
            logger.info(f"Matched {len(self.complete_flows)} complete flows")
            logger.info(f"Detected {len(self.split_flows)} split/incomplete flows")

    def _match_flow_packets(
        self, flow_info: FlowInfo, pcap_min: float | None, pcap_max: float | None
    ) -> list[tuple] | None:
        """
        Match packets to a flow using its time window.

        Searches for packets in both directions of the 5-tuple that fall within
        the flow's time window (timestamp to timestamp+duration). Direction is
        determined relative to the Flow ID (who initiated the connection).

        Returns:
            List of packets with direction markers: (timestamp, proto, size, flags, direction)
            where direction=0 means forward (matches Flow ID), direction=1 means backward.
            Returns None if no packets match.
        """
        # Flow ID defines forward direction (initiator -> responder)
        fwd_tuple = (
            flow_info.src_ip,
            flow_info.dst_ip,
            flow_info.src_port,
            flow_info.dst_port,
            flow_info.protocol,
        )
        # Reverse direction (responder -> initiator)
        bwd_tuple = (
            flow_info.dst_ip,
            flow_info.src_ip,
            flow_info.dst_port,
            flow_info.src_port,
            flow_info.protocol,
        )

        # Get packets for both directions
        fwd_packets = self.packets_by_tuple.get(fwd_tuple, [])
        bwd_packets = self.packets_by_tuple.get(bwd_tuple, [])

        if not fwd_packets and not bwd_packets:
            return None

        # Define time window from CSV
        flow_start = flow_info.timestamp
        flow_end = flow_start + flow_info.duration

        # Extract packets within time window and mark direction
        matched_fwd = [
            (ts, proto, size, flags, 0)  # 0 = forward (matches Flow ID direction)
            for ts, proto, size, flags in fwd_packets
            if flow_start <= ts <= flow_end
        ]
        matched_bwd = [
            (ts, proto, size, flags, 1)  # 1 = backward (opposite of Flow ID direction)
            for ts, proto, size, flags in bwd_packets
            if flow_start <= ts <= flow_end
        ]

        combined = matched_fwd + matched_bwd
        if not combined:
            return None

        # Sort by timestamp
        combined.sort(key=lambda pkt: pkt[0])

        # Check if flow is complete by comparing packet counts and boundaries
        actual_fwd = len(matched_fwd)
        actual_bwd = len(matched_bwd)
        expected_fwd = flow_info.fwd_packets
        expected_bwd = flow_info.bwd_packets

        # Detect split flows
        fwd_mismatch = actual_fwd != expected_fwd
        bwd_mismatch = actual_bwd != expected_bwd
        crosses_boundaries = (
            pcap_min is not None
            and pcap_max is not None
            and (flow_start < pcap_min or flow_end > pcap_max)
        )

        flow_info.is_complete = not (fwd_mismatch or bwd_mismatch or crosses_boundaries)

        return combined

    def _get_pcap_time_range(self) -> tuple[float | None, float | None]:
        """Determine the time range of packets in the current PCAP."""
        all_timestamps: list[float] = []

        for packets in self.packets_by_tuple.values():
            for pkt in packets:
                all_timestamps.append(pkt[0])

        if not all_timestamps:
            return None, None

        return min(all_timestamps), max(all_timestamps)

    def compute_features(self) -> pd.DataFrame:
        """Extract comprehensive ML features from complete flows only."""
        data_rows = []

        for _, (flow_info, packets) in tqdm(
            self.complete_flows.items(),
            desc="Extracting features",
            unit="flow",
            disable=not self.verbose,
        ):
            if not packets:
                continue

            # Compute flow-level features
            flow_duration = (
                packets[-1][0] - packets[0][0]
            )  # Last packet timestamp - first packet timestamp

            # Compute packet-level features
            packet_features = self._compute_packet_features(packets)

            # Create feature row
            row = (flow_info.protocol, flow_duration, *packet_features, flow_info.label)

            data_rows.append(row)

        # Create DataFrame
        df = self._create_features_dataframe(data_rows)

        if self.verbose:
            logger.info(f"Extracted features from {len(data_rows)} complete flows")
            logger.info(f"Feature matrix shape: {df.shape}")

        return df

    def _compute_packet_features(self, packets: list[tuple]) -> list[float]:
        """Extract packet-level features within sliding window."""
        # Create window of packets
        window = packets[: self.window_size]
        while len(window) < self.window_size:
            window.append((0.0, 0, 0, 0, 0))  # Padding

        packet_features = []
        for j, (timestamp, _, pkt_size, pkt_flags, direction) in enumerate(window):
            prev_timestamp = window[j - 1][0] if j > 0 and window[j - 1][0] != 0 else None
            pkt_iat = (
                timestamp - prev_timestamp if timestamp != 0 and prev_timestamp is not None else 0.0
            )
            packet_features.extend([pkt_size, pkt_flags, pkt_iat, direction])

        return packet_features

    def _create_features_dataframe(self, data_rows: list) -> pd.DataFrame:
        """Create properly formatted DataFrame with feature columns."""
        # Define column names
        base_columns = [
            "Flow Duration",
            "Protocol",
        ]

        packet_columns = []
        for i in range(1, self.window_size + 1):
            packet_columns.extend(
                [f"Pkt_Size{i}", f"Pkt_Flags{i}", f"Pkt_IAT{i}", f"Pkt_Direction{i}"]
            )

        columns = base_columns + packet_columns + ["Label"]

        df = pd.DataFrame(data_rows, columns=columns)

        # Convert numeric columns
        numeric_columns = ["Flow Duration"]
        numeric_columns.extend([f"Pkt_Size{i}" for i in range(1, self.window_size + 1)])
        numeric_columns.extend([f"Pkt_Flags{i}" for i in range(1, self.window_size + 1)])
        numeric_columns.extend([f"Pkt_IAT{i}" for i in range(1, self.window_size + 1)])
        numeric_columns.extend([f"Pkt_Direction{i}" for i in range(1, self.window_size + 1)])

        df[numeric_columns] = df[numeric_columns].astype(float)
        return df

    def save_output(self, df: pd.DataFrame, output_file: str | None = None) -> str:
        """Save extracted features to CSV file."""
        if output_file is None:
            output_file = os.path.splitext(self.pcap_file)[0] + "_features_with_labels.csv"

        # Select and reorder columns for output
        output_columns = (
            ["Flow Duration", "Protocol"]
            + [f"Pkt_Direction{i}" for i in range(1, self.window_size + 1)]
            + [f"Pkt_Flags{i}" for i in range(1, self.window_size + 1)]
            + [f"Pkt_IAT{i}" for i in range(1, self.window_size + 1)]
            + [f"Pkt_Size{i}" for i in range(1, self.window_size + 1)]
            + ["Label"]
        )

        output_df = df[output_columns]
        output_df.to_csv(output_file, index=False)

        if self.verbose:
            logger.info(f"Saved features to {output_file}")
            logger.info(f"Output shape: {output_df.shape}")

        return output_file

    def get_split_flow_report(self) -> pd.DataFrame:
        """Generate a report of split/incomplete flows for debugging."""
        split_data = []
        for flow_id, flow_info in self.split_flows.items():
            split_data.append(
                {
                    "Flow_ID": flow_id,
                    "Src_IP": flow_info.src_ip,
                    "Dst_IP": flow_info.dst_ip,
                    "Start_Time": pd.to_datetime(flow_info.timestamp, unit="s"),
                    "Duration": flow_info.duration,
                    "Expected_Packets": flow_info.total_packets,
                    "Label": flow_info.label,
                }
            )
        return pd.DataFrame(split_data)


def main():
    """Command-line interface for feature extraction."""

    args = parse_arguments()

    try:
        extractor = FeatureExtractor(
            pcap_file=args.pcap_file,
            labels_file=args.labels,
            window_size=args.window,
            verbose=not args.quiet,
        )

        extractor.process_packets()
        extractor.match_flows_with_labels()
        df = extractor.compute_features()
        output_file = extractor.save_output(df, args.output)

        if args.split_report:
            split_df = extractor.get_split_flow_report()
            split_df.to_csv(args.split_report, index=False)
            logger.info(f"Split flow report saved to {args.split_report}")

        print("Feature extraction completed successfully!")
        print(f"Output saved to: {output_file}")
        print(f"Complete flows: {len(extractor.complete_flows)}")
        print(f"Split flows: {len(extractor.split_flows)}")

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
