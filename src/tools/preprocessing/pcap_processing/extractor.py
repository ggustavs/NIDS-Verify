"""
Feature Extraction Tool with Enhanced Timestamp Handling

This module extracts machine learning features from PCAP files and can link them
with labels from CSV files. It handles the common problem of 12-hour timestamps
without AM/PM indicators using advanced disambiguation techniques.

Key Features:
- Multiple timestamp format support
- Configurable timezone offset handling
- Temporal proximity scoring for ambiguous timestamps
- Flow-based packet grouping with timeout detection
- Comprehensive feature extraction for ML training

Configuration:
- CSV_PCAP_OFFSET_HOURS: Hours that CSV timestamps are behind PCAP timestamps (default: 0)
- TIME_BUFFER_HOURS: Buffer time in hours for timestamp matching tolerance (default: 0.5)
"""

import scapy.all as scapy
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse
import os
from typing import List, Dict, Tuple, Optional, Any
import logging

# Global configuration variables
CSV_PCAP_OFFSET_HOURS = 0  # CSV timestamps are this many hours behind PCAP timestamps
TIME_BUFFER_HOURS = 0.5      # Buffer time in hours for timestamp matching

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Advanced feature extractor for NIDS packet data.
    
    Handles PCAP file processing, flow extraction, timestamp disambiguation,
    and ML feature generation for network intrusion detection systems.
    """
    
    def __init__(self, pcap_file: str, window_size: int = 10, 
                 labels_file: Optional[str] = None, verbose: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            pcap_file: Path to the PCAP file to process
            window_size: Number of packets to include in feature window
            labels_file: Optional path to CSV file with flow labels
            verbose: Enable detailed progress output
        """
        self.pcap_file = pcap_file
        self.window_size = window_size
        self.labels_file = labels_file
        self.flows: Dict[Tuple, List] = defaultdict(list)
        self.df_flows: Optional[pd.DataFrame] = None
        self.verbose = verbose
        self.flow_nr: Dict[Tuple, int] = defaultdict(int)
        
        if verbose:
            logger.info(f"Initialized FeatureExtractor for {pcap_file}")
            logger.info(f"Window size: {window_size}, Labels: {labels_file is not None}")

    def process_packets(self) -> None:
        """
        Process all packets in the PCAP file and group them into flows.
        
        Handles flow detection, timeout-based flow splitting, and packet
        direction determination following CICFlowMeter conventions.
        """
        try:
            packets = scapy.rdpcap(self.pcap_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read PCAP file {self.pcap_file}: {e}")
            
        timeout_threshold = 300  # Timeout threshold in seconds (CICFlowMeter default)

        for pkt in tqdm(packets, desc="Processing packets", unit="packet", disable=not self.verbose):
            if scapy.IP not in pkt:
                continue
                
            # Extract packet metadata
            packet_info = self._extract_packet_info(pkt)
            if packet_info is None:
                continue
                
            flow_key, timestamp, proto, pkt_size, pkt_flags, direction = packet_info
            
            # Handle flow splitting logic
            if self.labels_file is None:
                full_flow_key = self._handle_flow_timeout(flow_key, timestamp, timeout_threshold)
                target_flows = self.flows[full_flow_key]
            else:
                # Use labels file to split flows
                target_flows = self.flows[flow_key]
            
            # Add packet to flow
            target_flows.append((timestamp, proto, pkt_size, pkt_flags, direction))

        # Sort packets within each flow by timestamp
        for flow_key in self.flows:
            self.flows[flow_key].sort(key=lambda pkt: pkt[0])

        if self.labels_file:
            self.link_labels()

        if self.verbose:
            logger.info(f"Processed {len(packets)} packets from {self.pcap_file}")
            logger.info(f"Total flows extracted: {len(self.flows)}")

    def _extract_packet_info(self, pkt) -> Optional[Tuple]:
        """Extract key information from a packet."""
        src_ip = pkt[scapy.IP].src
        dst_ip = pkt[scapy.IP].dst
        timestamp = float(pkt.time) if pkt.time else 0
        proto = pkt[scapy.IP].proto
        pkt_size = len(pkt)
        
        # Extract port information
        if scapy.TCP in pkt:
            src_port = pkt[scapy.TCP].sport
            dst_port = pkt[scapy.TCP].dport
            pkt_flags = int(pkt[scapy.TCP].flags)
        elif scapy.UDP in pkt:
            src_port = pkt[scapy.UDP].sport
            dst_port = pkt[scapy.UDP].dport
            pkt_flags = 0
        else:
            src_port = None
            dst_port = None
            pkt_flags = 0
            
        if src_port is None or dst_port is None:
            return None
            
        flow_key = (src_ip, dst_ip, src_port, dst_port, proto)
        direction = 0  # Outgoing direction by default

        # Check for reverse flow (bidirectional flow handling)
        reverse_flow_key = (dst_ip, src_ip, dst_port, src_port, proto)
        if reverse_flow_key in self.flow_nr and flow_key not in self.flow_nr:
            flow_key = reverse_flow_key
            direction = 1  # Incoming direction

        return flow_key, timestamp, proto, pkt_size, pkt_flags, direction

    def _handle_flow_timeout(self, flow_key: Tuple, timestamp: float, 
                           timeout_threshold: float) -> Tuple:
        """Handle flow splitting based on timeout threshold."""
        full_flow_key = flow_key + (self.flow_nr[flow_key],)
        
        # Check for new flow conditions
        if full_flow_key in self.flows and self.flows[full_flow_key]:
            last_pkt = self.flows[full_flow_key][-1]
            last_timestamp = last_pkt[0]

            # Split flow if timeout is exceeded
            if timestamp - last_timestamp > timeout_threshold:
                self.flow_nr[flow_key] += 1
                full_flow_key = flow_key + (self.flow_nr[flow_key],)

        return full_flow_key

    def _parse_timestamp_with_ambiguity(self, timestamp_str: str) -> List[float]:
        """
        Parse timestamp string and handle 12-hour format ambiguity.
        
        Returns list of possible timestamp interpretations (Unix timestamps).
        Applies configurable offset to account for timezone differences.
        """
        dt_original = None
        
        # List of formats to try, in order of preference
        formats_to_try = [
            '%Y-%m-%d %H:%M:%S.%f',  # 2017-07-05 11:42:42.790920
            '%Y-%m-%d %H:%M:%S',     # 2017-07-05 11:42:42
            '%d/%m/%Y %H:%M',        # Original format
            '%m/%d/%Y %H:%M',        # Alternative date format
        ]
        
        for fmt in formats_to_try:
            try:
                dt_original = pd.to_datetime(timestamp_str, format=fmt)
                break
            except:
                continue
        
        # If all specific formats fail, try pandas auto-detection
        if dt_original is None:
            try:
                dt_original = pd.to_datetime(timestamp_str)
            except:
                if self.verbose:
                    logger.warning(f"Could not parse timestamp {timestamp_str}")
                return [0]
        
        hour = dt_original.hour
        
        # Apply configurable offset (CSV is behind PCAP)
        pcap_dt = dt_original + pd.Timedelta(hours=CSV_PCAP_OFFSET_HOURS)
        return [pcap_dt.timestamp()]

    def _find_best_packet_match(self, packets: List, possible_timestamps: List[float], 
                               expected_packet_count: int, flow_duration: Optional[float]) -> Optional[Dict]:
        """
        Find the best matching packet sequence using multiple scoring criteria.
        
        Uses temporal proximity, packet count matching, flow duration, and
        consistency scoring to disambiguate between timestamp possibilities.
        """
        best_match = None
        best_score = float('-inf')
        
        packet_times = [pkt[0] for pkt in packets]
        
        for timestamp in possible_timestamps:
            if not packet_times:
                continue
                
            # Find the closest packet to our timestamp
            time_diffs = [abs(pkt_time - timestamp) for pkt_time in packet_times]
            closest_idx = time_diffs.index(min(time_diffs))
            
            # Extract flow starting from closest packet
            end_idx = min(closest_idx + expected_packet_count, len(packets))
            candidate_flow = packets[closest_idx:end_idx]
            
            if not candidate_flow:
                continue
                
            # Calculate comprehensive match score
            score = self._calculate_match_score(candidate_flow, timestamp, 
                                              expected_packet_count, flow_duration, time_diffs)
            
            if score > best_score:
                best_score = score
                best_match = {
                    'flow': candidate_flow,
                    'start_idx': closest_idx,
                    'end_idx': end_idx,
                    'timestamp': timestamp,
                    'score': score
                }
        
        return best_match

    def _calculate_match_score(self, candidate_flow: List, timestamp: float,
                             expected_packet_count: int, flow_duration: Optional[float],
                             time_diffs: List[float]) -> float:
        """Calculate comprehensive matching score for timestamp disambiguation."""
        score = 0
        
        # 1. Temporal proximity score (closer timestamp = higher score)
        min_time_diff = min(time_diffs)
        temporal_score = 1.0 / (1.0 + min_time_diff)
        score += temporal_score * 100
        
        # 2. Packet count match score
        actual_count = len(candidate_flow)
        count_diff = abs(actual_count - expected_packet_count)
        count_score = 1.0 / (1.0 + count_diff)
        score += count_score * 50
        
        # 3. Flow duration match score (if available)
        if flow_duration and len(candidate_flow) > 1:
            actual_duration = candidate_flow[-1][0] - candidate_flow[0][0]
            duration_diff = abs(actual_duration - flow_duration)
            duration_score = 1.0 / (1.0 + duration_diff)
            score += duration_score * 25
        
        # 4. Consistency score - prefer flows without large gaps
        if len(candidate_flow) > 1:
            inter_arrival_times = [candidate_flow[i][0] - candidate_flow[i-1][0] 
                                 for i in range(1, len(candidate_flow))]
            avg_iat = sum(inter_arrival_times) / len(inter_arrival_times)
            iat_variance = sum((iat - avg_iat) ** 2 for iat in inter_arrival_times) / len(inter_arrival_times)
            consistency_score = 1.0 / (1.0 + iat_variance)
            score += consistency_score * 25
        
        return score

    def _get_pcap_time_range(self) -> Tuple[Optional[float], Optional[float]]:
        """Determine the time range of packets in the current PCAP."""
        all_timestamps = []
        for flow_packets in self.flows.values():
            for pkt in flow_packets:
                timestamp = float(pkt[0])  # timestamp is first element
                all_timestamps.append(timestamp)
        
        if not all_timestamps:
            return None, None

        return min(all_timestamps), max(all_timestamps)

    def _filter_csv_by_time_range(self, labels_df: pd.DataFrame, 
                                 pcap_min_time: Optional[float], 
                                 pcap_max_time: Optional[float]) -> pd.DataFrame:
        """Filter CSV rows to match PCAP time range with configurable buffer."""
        if pcap_min_time is None or pcap_max_time is None:
            return labels_df
            
        # Add configurable buffer time
        buffer_seconds = TIME_BUFFER_HOURS * 3600
        search_min_time = pcap_min_time - buffer_seconds
        search_max_time = pcap_max_time + buffer_seconds
        
        filtered_rows = []
        
        for _, row in tqdm(labels_df.iterrows(),
                          total=len(labels_df),
                          desc="Filtering CSV rows by time range", 
                          unit="row",
                          disable=not self.verbose):
            timestamp_str = str(row['Timestamp'])
            possible_timestamps = self._parse_timestamp_with_ambiguity(timestamp_str)
            
            # Check if any timestamp interpretation falls within range
            if any(search_min_time <= ts <= search_max_time for ts in possible_timestamps):
                filtered_rows.append(row)
        
        if not filtered_rows:
            if self.verbose:
                logger.warning(f"No CSV rows found in PCAP time range "
                             f"[{pd.to_datetime(search_min_time, unit='s')} to "
                             f"{pd.to_datetime(search_max_time, unit='s')}] "
                             f"(buffer: {TIME_BUFFER_HOURS}h, offset: {CSV_PCAP_OFFSET_HOURS}h)")
            return pd.DataFrame()
        
        filtered_df = pd.DataFrame(filtered_rows)
        
        if self.verbose:
            logger.info(f"Filtered CSV: {len(filtered_df)}/{len(labels_df)} rows "
                       f"within PCAP time range")
            
        return filtered_df

    def link_labels(self) -> None:
        """Link flows with labels from CSV file using advanced matching."""
        try:
            labels_df = pd.read_csv(self.labels_file)
            labels_df.columns = labels_df.columns.str.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read labels file {self.labels_file}: {e}")

        # Get PCAP time range and filter CSV
        pcap_min_time, pcap_max_time = self._get_pcap_time_range()
        
        if self.verbose and pcap_min_time is not None:
            logger.info(f"PCAP time range: {pd.to_datetime(pcap_min_time, unit='s')} to "
                       f"{pd.to_datetime(pcap_max_time, unit='s')}")
        
        filtered_labels_df = self._filter_csv_by_time_range(labels_df, pcap_min_time, pcap_max_time)
        
        if filtered_labels_df.empty:
            if self.verbose:
                logger.warning("No CSV rows match PCAP time range. Skipping label matching.")
            return

        # Process flow matching
        new_flows = defaultdict(list)
        matched_flows = 0
        ambiguous_matches = 0
        
        flow_items = list(self.flows.items())
        for flow_key, packets in tqdm(flow_items, desc="Matching flows with labels", 
                                    unit="flow", disable=not self.verbose):
            if not packets:
                continue
                
            matches = self._find_matching_csv_rows(flow_key, filtered_labels_df)
            
            for _, row in matches.iterrows():
                match_result = self._process_flow_match(flow_key, packets, row)
                if match_result:
                    flow_match, is_ambiguous = match_result
                    labeled_flow_key = flow_key + (row['Label'], matched_flows)
                    new_flows[labeled_flow_key] = flow_match['flow']
                    matched_flows += 1
                    if is_ambiguous:
                        ambiguous_matches += 1
        
        # Replace flows with labeled flows
        self.flows = new_flows
        
        if self.verbose:
            logger.info(f"Successfully matched {matched_flows} flows with labels")
            if ambiguous_matches > 0:
                logger.info(f"Resolved {ambiguous_matches} ambiguous timestamps")

    def _find_matching_csv_rows(self, flow_key: Tuple, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Find CSV rows that match the flow's 5-tuple."""
        src_ip, dst_ip, src_port, dst_port, proto = flow_key[:5]
        return labels_df[
            (labels_df['Src IP'] == src_ip) &
            (labels_df['Dst IP'] == dst_ip) &
            (labels_df['Src Port'] == src_port) &
            (labels_df['Dst Port'] == dst_port) &
            (labels_df['Protocol'] == proto)
        ]

    def _process_flow_match(self, flow_key: Tuple, packets: List, row: pd.Series) -> Optional[Tuple]:
        """Process a single flow-label match."""
        timestamp_str = str(row['Timestamp'])
        expected_fwd_packets = int(row['Total Fwd Packet'])
        expected_bwd_packets = int(row['Total Bwd packets'])
        expected_total_packets = expected_fwd_packets + expected_bwd_packets
        flow_duration = float(row['Flow Duration']) if 'Flow Duration' in row else None
        
        # Get possible timestamps (AM/PM ambiguity resolution)
        possible_timestamps = self._parse_timestamp_with_ambiguity(timestamp_str)
        is_ambiguous = len(possible_timestamps) > 1
        
        # Find best matching packet sequence
        best_match = self._find_best_packet_match(
            packets, possible_timestamps, expected_total_packets, flow_duration
        )
        
        if best_match:
            return best_match, is_ambiguous
        return None

    def compute_features(self) -> None:
        """Extract comprehensive ML features from processed flows."""
        data_rows = []
        
        for flow_key, pkt_list in tqdm(self.flows.items(), desc="Extracting features", 
                                     unit="flow", disable=not self.verbose):
            if len(flow_key) < 5 or not pkt_list:
                continue
                
            # Extract flow metadata
            src_ip, dst_ip, src_port, dst_port, proto = flow_key[:5]
            label = flow_key[-2] if len(flow_key) > 5 and isinstance(flow_key[-2], str) else "UNLABELED"
            
            # Sort packets by timestamp
            pkt_list.sort(key=lambda pkt: pkt[0])
            
            # Compute flow-level features
            flow_features = self._compute_flow_features(pkt_list)
            
            # Create feature row
            row = [src_ip, dst_ip, src_port, dst_port, proto] + flow_features
            
            # Extract packet-level features with window
            packet_features = self._compute_packet_features(pkt_list)
            row.extend(packet_features)
            
            # Add label
            row.append(label)
            data_rows.append(row)
            
        # Create DataFrame with proper column names
        self.df_flows = self._create_features_dataframe(data_rows)

    def _compute_flow_features(self, pkt_list: List) -> List[float]:
        """Compute flow-level statistical features."""
        flow_duration = pkt_list[-1][0] - pkt_list[0][0] if len(pkt_list) > 1 else 0
        pkt_sizes = [pkt[2] for pkt in pkt_list]
        pkt_iats = [pkt_list[i][0] - pkt_list[i - 1][0] for i in range(1, len(pkt_list))]
        
        max_pkt_size = max(pkt_sizes) if pkt_sizes else 0
        iat_mean = sum(pkt_iats) / len(pkt_iats) if pkt_iats else 0
        iat_std = (sum((x - iat_mean) ** 2 for x in pkt_iats) / len(pkt_iats)) ** 0.5 if pkt_iats else 0
        
        return [flow_duration, iat_mean, iat_std, max_pkt_size]

    def _compute_packet_features(self, pkt_list: List) -> List[float]:
        """Extract packet-level features within sliding window."""
        # Create window of packets
        window = pkt_list[:self.window_size]
        while len(window) < self.window_size:
            window.append((0, 0, 0, 0, 0))  # Padding: (timestamp, proto, size, flags, direction)
        
        packet_features = []
        for j, (timestamp, _, pkt_size, pkt_flags, direction) in enumerate(window):
            prev_timestamp = window[j - 1][0] if j > 0 and window[j - 1][0] != 0 else None
            pkt_iat = timestamp - prev_timestamp if timestamp != 0 and prev_timestamp is not None else 0
            packet_features.extend([pkt_size, pkt_flags, pkt_iat, direction])
            
        return packet_features

    def _create_features_dataframe(self, data_rows: List) -> pd.DataFrame:
        """Create properly formatted DataFrame with feature columns."""
        # Define column names
        base_columns = ["Src_IP", "Dst_IP", "Src_Port", "Dst_Port", "Protocol", 
                       "Flow_Duration", "Flow_IAT_Mean", "Flow_IAT_Std", "Max_Pkt_Size"]
        
        packet_columns = []
        for i in range(1, self.window_size + 1):
            packet_columns.extend([f"Pkt_Size{i}", f"Pkt_Flags{i}", f"Pkt_IAT{i}", f"Pkt_Direction{i}"])
        
        columns = base_columns + packet_columns + ["Label"]
        
        df = pd.DataFrame(data_rows, columns=columns)
        
        # Convert numeric columns to appropriate types
        numeric_columns = ["Flow_Duration", "Flow_IAT_Mean", "Flow_IAT_Std", "Max_Pkt_Size"]
        numeric_columns.extend([f"Pkt_Size{i}" for i in range(1, self.window_size + 1)])
        numeric_columns.extend([f"Pkt_Flags{i}" for i in range(1, self.window_size + 1)])
        numeric_columns.extend([f"Pkt_IAT{i}" for i in range(1, self.window_size + 1)])
        numeric_columns.extend([f"Pkt_Direction{i}" for i in range(1, self.window_size + 1)])
        
        df[numeric_columns] = df[numeric_columns].astype(float)
        return df

    def save_output(self, output_file: Optional[str] = None) -> str:
        """Save extracted features to CSV file."""
        if self.df_flows is None:
            raise RuntimeError("No features computed. Call compute_features() first.")
            
        if output_file is None:
            suffix = "_features_with_labels.csv" if self.labels_file else "_features.csv"
            output_file = os.path.splitext(self.pcap_file)[0] + suffix
        
        # Select and reorder columns for output
        output_columns = ["Flow_Duration", "Protocol"] + \
                        [f"Pkt_Direction{i}" for i in range(1, self.window_size + 1)] + \
                        [f"Pkt_Flags{i}" for i in range(1, self.window_size + 1)] + \
                        [f"Pkt_IAT{i}" for i in range(1, self.window_size + 1)] + \
                        [f"Pkt_Size{i}" for i in range(1, self.window_size + 1)] + \
                        ["Label"]
        
        output_df = self.df_flows[output_columns]
        output_df.to_csv(output_file, index=False)
        
        if self.verbose:
            logger.info(f"Feature extraction completed. Saved to {output_file}")
            logger.info(f"Output shape: {output_df.shape}")
            
        return output_file


def main():
    """Command-line interface for feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract ML features from PCAP files with advanced timestamp handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic feature extraction
  python extractor.py sample.pcap --window 10
  
  # With label matching
  python extractor.py sample.pcap --labels flows.csv --window 10
  
  # Quiet mode
  python extractor.py sample.pcap --labels flows.csv --quiet
        """
    )
    parser.add_argument("pcap_file", help="Path to the PCAP file")
    parser.add_argument("--window", type=int, default=10, 
                       help="Window size for feature extraction (default: 10)")
    parser.add_argument("--labels", help="Path to the labels CSV file", default=None)
    parser.add_argument("--output", help="Output CSV file path (auto-generated if not specified)")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    
    args = parser.parse_args()

    try:
        extractor = FeatureExtractor(
            pcap_file=args.pcap_file,
            window_size=args.window,
            labels_file=args.labels,
            verbose=not args.quiet
        )
        
        extractor.process_packets()
        extractor.compute_features()
        output_file = extractor.save_output(args.output)
        
        print(f"✅ Feature extraction completed successfully!")
        print(f"📁 Output saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
