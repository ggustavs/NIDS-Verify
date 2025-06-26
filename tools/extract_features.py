"""
Feature Extraction Tool with 12-Hour Timestamp Ambiguity Resolution

This tool extracts machine learning features from PCAP files and can link them
with labels from CSV files. It handles the common problem of 12-hour timestamps
without AM/PM indicators by:

1. Generating both AM and PM possibilities for ambiguous timestamps
2. Applying a configurable offset to account for timezone differences (CSV timestamps behind PCAP)
3. Filtering CSV rows to only match those within the PCAP split's time range (for split PCAP files)
4. Using multiple scoring criteria to find the best match:
   - Temporal proximity (how close the packet timestamp is to the CSV timestamp)
   - Packet count matching (expected vs actual packet count)
   - Flow duration matching (if available in CSV)
   - Inter-arrival time consistency (flows with consistent timing patterns)

The scoring system weights these factors to automatically select the most
likely correct interpretation of ambiguous timestamps. For split PCAP files,
only CSV rows with timestamps that could potentially match the current split
are considered, improving performance and accuracy.

Configuration:
- CSV_PCAP_OFFSET_HOURS: Hours that CSV timestamps are behind PCAP timestamps (default: 3)
- TIME_BUFFER_HOURS: Buffer time in hours for timestamp matching tolerance (default: 1)

Usage:
    python extract_features.py input.pcap --labels flows.csv --window 10
"""

import scapy.all as scapy
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse
import os

# Global configuration variables
CSV_PCAP_OFFSET_HOURS = 3  # CSV timestamps are this many hours behind PCAP timestamps
TIME_BUFFER_HOURS = 1      # Buffer time in hours for timestamp matching

class FeatureExtractor:
    def __init__(self, pcap_file, window_size=10, labels_file=None, verbose=True):
        self.pcap_file = pcap_file
        self.window_size = window_size
        self.labels_file = labels_file
        self.flows = defaultdict(list)  # Store flows as a dictionary of lists
        self.df_flows = None
        self.verbose = verbose
        self.flow_nr = defaultdict(int)

    def process_packets(self):
        packets = scapy.rdpcap(self.pcap_file)
        timeout_threshold = 300  # Timeout threshold in seconds (CICFlowMeter default)

        for pkt in tqdm(packets, desc="Processing packets", unit="packet"):
            if scapy.IP in pkt:
                src_ip = pkt[scapy.IP].src
                dst_ip = pkt[scapy.IP].dst
                timestamp = float(pkt.time) if pkt.time else 0  # Convert to float
                proto = pkt[scapy.IP].proto
                pkt_size = len(pkt)
                if scapy.TCP in pkt:
                    src_port = pkt[scapy.TCP].sport
                    dst_port = pkt[scapy.TCP].dport
                elif scapy.UDP in pkt:
                    src_port = pkt[scapy.UDP].sport
                    dst_port = pkt[scapy.UDP].dport
                else:
                    src_port = None
                    dst_port = None
                flow_key = (src_ip, dst_ip, src_port, dst_port, proto)  # Match CICFlowMeter's flow key
                direction = 0  # Outgoing direction by default

                if flow_key not in self.flow_nr:
                    reverse_flow_key = (dst_ip, src_ip, dst_port, src_port, proto)
                    if reverse_flow_key in self.flow_nr:
                        flow_key = reverse_flow_key
                        direction = 1  # Set to incoming direction 
                else:
                    self.flow_nr[flow_key] == 0
            

                if self.labels_file is None:
                    full_flow_key = flow_key + (self.flow_nr[flow_key],)  # Add flow number to the key
                    # Check for new flow conditions
                    last_pkt = self.flows[full_flow_key][-1] if self.flows[full_flow_key] else None
                    if last_pkt:
                        last_timestamp = last_pkt[0]

                        # Split flow if timeout or sequence number difference is exceeded
                        if timestamp - last_timestamp > timeout_threshold:
                            self.flow_nr[flow_key] += 1
                            full_flow_key = (src_ip, dst_ip, src_port, dst_port, proto, self.flow_nr[flow_key])  # Create a new flow key

                    pkt_flags = int(pkt[scapy.TCP].flags) if scapy.TCP in pkt else 0
                    self.flows[full_flow_key].append((timestamp, proto, pkt_size, pkt_flags, direction))
                else:
                    # We use the labels file to split flows
                    pkt_flags = int(pkt[scapy.TCP].flags) if scapy.TCP in pkt else 0
                    self.flows[flow_key].append((timestamp, proto, pkt_size, pkt_flags, direction))

        # Sort packets within each flow
        for flow_key in self.flows:
            self.flows[flow_key].sort(key=lambda pkt: pkt[0])  # Sort by timestamp

        if self.labels_file:
            self.link_labels()

        if self.verbose:
            print(f"Processed {len(packets)} packets from {self.pcap_file}")
            print(f"Total flows extracted: {len(self.flows)}")
        
    def _parse_timestamp_with_ambiguity(self, timestamp_str):
        """
        Parse timestamp string and return both AM and PM possibilities.
        Handles the 12-hour format ambiguity by returning both possible interpretations.
        Also applies a configurable offset to account for timezone differences between CSV and PCAP.
        """
        
        # Try to parse the timestamp - try both M/D/Y and D/M/Y formats
        dt_original = None
        try:
            dt_original = pd.to_datetime(timestamp_str, format='%d/%m/%Y %H:%M')
        except:
            try:
                # Fallback: let pandas auto-detect
                dt_original = pd.to_datetime(timestamp_str)
            except:
                if self.verbose:
                    print(f"Warning: Could not parse timestamp {timestamp_str}")
                return [0]
        
        hour = dt_original.hour
        
        # If hour is <= 12, create both AM and PM possibilities
        if hour <= 12:
            # AM possibility
            if hour == 12:
                # 12:XX AM = midnight (00:XX)
                am_dt = dt_original.replace(hour=0)
            else:
                # Regular AM hour (keep as is)
                am_dt = dt_original
            
            # PM possibility  
            if hour == 12:
                # 12:XX PM = noon (keep 12:XX)
                pm_dt = dt_original
            else:
                # Regular PM hour (add 12 hours)
                pm_dt = dt_original + pd.Timedelta(hours=12)
            
            # Apply configurable offset (CSV is behind PCAP)
            am_pcap = am_dt + pd.Timedelta(hours=CSV_PCAP_OFFSET_HOURS)
            pm_pcap = pm_dt + pd.Timedelta(hours=CSV_PCAP_OFFSET_HOURS)
            
            return [am_pcap.timestamp(), pm_pcap.timestamp()]
        else:
            # Hour > 12, so it's already in 24-hour format, just apply offset
            pcap_dt = dt_original + pd.Timedelta(hours=CSV_PCAP_OFFSET_HOURS)
            return [pcap_dt.timestamp()]

    def _find_best_packet_match(self, packets, possible_timestamps, expected_packet_count, flow_duration):
        """
        Find the best matching packet sequence for given timestamps and expected packet count.
        Uses multiple criteria to disambiguate between AM/PM possibilities.
        """
        best_match = None
        best_score = float('-inf')
        
            
        packet_times = [pkt[0] for pkt in packets]
        
        for timestamp in possible_timestamps:
            # Find packets around this timestamp
            
            # Find the closest packet to our timestamp
            time_diffs = [abs(pkt_time - timestamp) for pkt_time in packet_times]
            if not time_diffs:
                continue
                
            closest_idx = time_diffs.index(min(time_diffs))
            
            # Extract flow starting from closest packet
            end_idx = min(closest_idx + expected_packet_count, len(packets))
            candidate_flow = packets[closest_idx:end_idx]
            
            if not candidate_flow:
                continue
                
            # Calculate match score based on multiple criteria
            score = 0
            
            # 1. Temporal proximity score (closer timestamp = higher score)
            min_time_diff = min(time_diffs)
            temporal_score = 1.0 / (1.0 + min_time_diff)  # Higher score for closer timestamps
            score += temporal_score * 100
            
            # 2. Packet count match score
            actual_count = len(candidate_flow)
            count_diff = abs(actual_count - expected_packet_count)
            count_score = 1.0 / (1.0 + count_diff)  # Higher score for closer packet counts
            score += count_score * 50
            
            # 3. Flow duration match score (if flow_duration is available)
            if flow_duration and len(candidate_flow) > 1:
                actual_duration = candidate_flow[-1][0] - candidate_flow[0][0]
                duration_diff = abs(actual_duration - flow_duration)
                duration_score = 1.0 / (1.0 + duration_diff)
                score += duration_score * 25
            
            # 4. Consistency score - prefer flows that don't have large gaps
            if len(candidate_flow) > 1:
                inter_arrival_times = [candidate_flow[i][0] - candidate_flow[i-1][0] 
                                     for i in range(1, len(candidate_flow))]
                avg_iat = sum(inter_arrival_times) / len(inter_arrival_times)
                iat_variance = sum((iat - avg_iat) ** 2 for iat in inter_arrival_times) / len(inter_arrival_times)
                consistency_score = 1.0 / (1.0 + iat_variance)
                score += consistency_score * 25
            
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

    def _get_pcap_time_range(self): # TODO: Fix this inefficient method
        """
        Determine the time range of packets in the current PCAP split.
        Returns (min_time, max_time) of all packets in all flows.
        """
        all_timestamps = []
        for flow_packets in self.flows.values():
            for pkt in flow_packets:
                # Handle Decimal timestamps from scapy
                timestamp = float(pkt[0])  # timestamp is first element
                all_timestamps.append(timestamp)
        
        if not all_timestamps:
            return None, None

        min_time = float(min(all_timestamps))
        max_time = float(max(all_timestamps))

        return min_time, max_time

    def _filter_csv_by_time_range(self, labels_df, pcap_min_time, pcap_max_time):
        """
        Filter CSV rows to only include those that could potentially match
        packets in the current PCAP split's time range.
        Accounts for configurable offset and AM/PM ambiguity.
        """
        if pcap_min_time is None or pcap_max_time is None:
            return labels_df  # Return all rows if no time range available
            
        # Add configurable buffer time to account for potential timing variations
        buffer_seconds = TIME_BUFFER_HOURS * 3600
        search_min_time = pcap_min_time - buffer_seconds
        search_max_time = pcap_max_time + buffer_seconds
        
        filtered_rows = []
        
        # Process CSV rows with progress bar
        for _, row in tqdm(labels_df.iterrows(), 
                          desc="Filtering CSV rows by time range", 
                          unit="row"):
            timestamp_str = str(row['Timestamp'])
            possible_timestamps = self._parse_timestamp_with_ambiguity(timestamp_str)
            
            # Check if any of the possible timestamp interpretations fall within our search range
            timestamp_in_range = False
            for ts in possible_timestamps:
                if search_min_time <= ts <= search_max_time:
                    timestamp_in_range = True
                    break
            
            if timestamp_in_range:
                filtered_rows.append(row)
        
        if not filtered_rows:
            if self.verbose:
                print(f"Warning: No CSV rows found in PCAP time range "
                      f"[{pd.to_datetime(search_min_time, unit='s')} to "
                      f"{pd.to_datetime(search_max_time, unit='s')}] "
                      f"(using {TIME_BUFFER_HOURS}h buffer and {CSV_PCAP_OFFSET_HOURS}h offset)")
            return pd.DataFrame()  # Return empty DataFrame
        
        filtered_df = pd.DataFrame(filtered_rows)
        
        if self.verbose:
            print(f"Filtered CSV: {len(filtered_df)} rows (out of {len(labels_df)}) "
                  f"fall within PCAP time range (offset: {CSV_PCAP_OFFSET_HOURS}h, buffer: {TIME_BUFFER_HOURS}h)")
            
        return filtered_df

    def link_labels(self):
        labels_df = pd.read_csv(self.labels_file)
        labels_df.columns = labels_df.columns.str.strip()  # Strip whitespace from column names

        # Get the time range of the current PCAP split
        pcap_min_time, pcap_max_time = self._get_pcap_time_range()
        
        if self.verbose and pcap_min_time is not None:
            print(f"PCAP time range: {pd.to_datetime(pcap_min_time, unit='s')} to "
                  f"{pd.to_datetime(pcap_max_time, unit='s')}")
        
        # Filter CSV rows to only those that could potentially match this PCAP split
        filtered_labels_df = self._filter_csv_by_time_range(labels_df, pcap_min_time, pcap_max_time)
        
        if filtered_labels_df.empty:
            if self.verbose:
                print("No CSV rows match the PCAP time range. Skipping label matching.")
            return

        new_flows = defaultdict(list)  # New dictionary to hold flows with labels
        matched_flows = 0
        ambiguous_matches = 0
        
        # Process flows with progress bar
        flow_items = list(self.flows.items())
        for flow_key, packets in tqdm(flow_items, desc="Matching flows with labels", unit="flow"):
            if not packets:
                continue
                
            src_ip, dst_ip, src_port, dst_port, proto = flow_key[:5]
            matching_rows = filtered_labels_df[(filtered_labels_df['Source IP'] == src_ip) &
                                             (filtered_labels_df['Destination IP'] == dst_ip) &
                                             (filtered_labels_df['Source Port'] == src_port) &
                                             (filtered_labels_df['Destination Port'] == dst_port) &
                                             (filtered_labels_df['Protocol'] == proto)]
            
            if not matching_rows.empty:
                for _, row in matching_rows.iterrows():
                    timestamp_str = str(row['Timestamp'])
                    label = row['Label']
                    expected_fwd_packets = int(row['Total Fwd Packets'])
                    expected_bwd_packets = int(row['Total Backward Packets'])
                    expected_total_packets = expected_fwd_packets + expected_bwd_packets
                    flow_duration = float(row['Flow Duration']) if 'Flow Duration' in row else None
                    
                    # Get possible timestamps (AM/PM ambiguity resolution)
                    possible_timestamps = self._parse_timestamp_with_ambiguity(timestamp_str)
                    
                    if len(possible_timestamps) > 1:
                        ambiguous_matches += 1
                    
                    # Find best matching packet sequence
                    best_match = self._find_best_packet_match(
                        packets, possible_timestamps, expected_total_packets, flow_duration
                    )
                    
                    if best_match:
                        # Create new flow key with label
                        labeled_flow_key = flow_key + (label, matched_flows)
                        new_flows[labeled_flow_key] = best_match['flow']
                        matched_flows += 1
        
        # Replace the original flows with labeled flows
        self.flows = new_flows
        
        if self.verbose:
            print(f"Successfully matched {matched_flows} flows with labels")
            if ambiguous_matches > 0:
                print(f"Resolved {ambiguous_matches} ambiguous timestamps (offset: {CSV_PCAP_OFFSET_HOURS}h, buffer: {TIME_BUFFER_HOURS}h)")
        
        return
        

    def compute_features(self):
        data_rows = []
        for flow_key, pkt_list in tqdm(self.flows.items(), desc="Extracting features", unit="flow"):
            # Handle different flow key structures (with or without labels)
            if len(flow_key) >= 5:
                src_ip, dst_ip, src_port, dst_port, proto = flow_key[:5]
            else:
                continue  # Skip malformed flow keys
            
            # Extract label if available
            label = flow_key[-2] if len(flow_key) > 5 and isinstance(flow_key[-2], str) else "UNLABELED"
            
            # Sort packets by timestamp
            pkt_list.sort(key=lambda pkt: pkt[0])
            
            if not pkt_list:
                continue
                
            flow_duration = pkt_list[-1][0] - pkt_list[0][0] if len(pkt_list) > 1 else 0
            pkt_sizes = [pkt[2] for pkt in pkt_list]
            pkt_iats = [pkt_list[i][0] - pkt_list[i - 1][0] for i in range(1, len(pkt_list))]
            max_pkt_size = max(pkt_sizes) if pkt_sizes else 0
            iat_mean = sum(pkt_iats) / len(pkt_iats) if pkt_iats else 0
            iat_std = (sum((x - iat_mean) ** 2 for x in pkt_iats) / len(pkt_iats)) ** 0.5 if pkt_iats else 0
            
            row = [src_ip, dst_ip, src_port, dst_port, proto, flow_duration, iat_mean, iat_std, max_pkt_size]
            
            # Create window of packets
            window = pkt_list[:self.window_size]
            while len(window) < self.window_size:
                window.append((0, 0, 0, 0, 0))  # Add placeholders: (timestamp, proto, pkt_size, pkt_flags, direction)
            
            # Extract packet-level features
            for j, (timestamp, _, pkt_size, pkt_flags, direction) in enumerate(window):
                prev_timestamp = window[j - 1][0] if j > 0 and window[j - 1][0] != 0 else None
                pkt_iat = timestamp - prev_timestamp if timestamp != 0 and prev_timestamp is not None else 0
                row.extend([pkt_size, pkt_flags, pkt_iat, direction])
            
            # Add label at the end
            row.append(label)
            data_rows.append(row)
            
        # Create column names
        columns = ["Source_IP", "Destination_IP", "Source_Port", "Destination_Port", "Protocol", 
                  "Flow_Duration", "Flow_IAT_Mean", "Flow_IAT_Std", "Max_Pkt_Size"]
        for i in range(1, self.window_size + 1):
            columns.extend([f"Pkt_Size{i}", f"Pkt_Flags{i}", f"Pkt_IAT{i}", f"Pkt_Direction{i}"])
        columns.append("Label")
        
        self.df_flows = pd.DataFrame(data_rows, columns=columns)
        
        # Convert numeric columns to appropriate types
        numeric_columns = ["Flow_Duration", "Flow_IAT_Mean", "Flow_IAT_Std", "Max_Pkt_Size"]
        numeric_columns.extend([f"Pkt_Size{i}" for i in range(1, self.window_size + 1)])
        numeric_columns.extend([f"Pkt_Flags{i}" for i in range(1, self.window_size + 1)])
        numeric_columns.extend([f"Pkt_IAT{i}" for i in range(1, self.window_size + 1)])
        numeric_columns.extend([f"Pkt_Direction{i}" for i in range(1, self.window_size + 1)])
        
        self.df_flows[numeric_columns] = self.df_flows[numeric_columns].astype(float)

    def save_output(self):
        output_file = os.path.splitext(self.pcap_file)[0] + "_features_with_labels.csv" if self.labels_file else os.path.splitext(self.pcap_file)[0] + "_features.csv"
        output_df = self.df_flows.drop(columns=["Flow_Duration", "Flow_IAT_Mean", "Flow_IAT_Std", "Max_Pkt_Size"], errors='ignore')
        output_df.to_csv(output_file, index=False)
        if self.verbose:
            print(f"Feature extraction completed. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ML features from PCAP files")
    parser.add_argument("pcap_file", help="Path to the PCAP file")
    parser.add_argument("--window", type=int, default=10, help="Window size for feature extraction")
    parser.add_argument("--labels", help="Path to the labels CSV file", default=None)
    # Parser is verbose by default, add argument to disable it
    parser.add_argument("--not-verbose", action="store_false", help="Disable verbose output")
    args = parser.parse_args()

    extractor = FeatureExtractor(args.pcap_file, args.window, args.labels, args.not_verbose)
    extractor.process_packets()
    extractor.compute_features()
    extractor.save_output()
