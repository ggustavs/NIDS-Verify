import scapy.all as scapy
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse
import os
from sklearn.preprocessing import MinMaxScaler

class FeatureExtractor:
    def __init__(self, pcap_file, window_size=10, labels_file=None, verbose=True):
        self.pcap_file = pcap_file
        self.window_size = window_size
        self.labels_file = labels_file
        self.flows = {}
        self.df_flows = None
        self.verbose = verbose

    def process_packets(self):
        packets = scapy.rdpcap(self.pcap_file)
        timeout_threshold = 300  # Timeout threshold in seconds (CICFlowMeter default)
        seq_diff_threshold = 500000  # Sequence number difference threshold

        for pkt in tqdm(packets, desc="Processing packets", unit="packet"):
            if scapy.IP in pkt:
                src_ip = pkt[scapy.IP].src
                dst_ip = pkt[scapy.IP].dst
                timestamp = pkt.time if pkt.time else 0
                proto = pkt[scapy.IP].proto
                pkt_size = len(pkt)
                src_port = pkt[scapy.TCP].sport if scapy.TCP in pkt else (pkt[scapy.UDP].sport if scapy.UDP in pkt else None)
                dst_port = pkt[scapy.TCP].dport if scapy.TCP in pkt else (pkt[scapy.UDP].dport if scapy.UDP in pkt else None)
                seq_num = pkt[scapy.TCP].seq if scapy.TCP in pkt else None
                flow_key = (src_ip, dst_ip, src_port, dst_port, proto)  # Match CICFlowMeter's flow key

                if flow_key not in self.flows:
                    self.flows[flow_key] = []

                # Check for new flow conditions
                last_pkt = self.flows[flow_key][-1] if self.flows[flow_key] else None
                if last_pkt:
                    last_timestamp = last_pkt[0]
                    last_seq_num = last_pkt[5]
                    last_flags = last_pkt[3]

                    # Split flow if timeout or sequence number difference is exceeded
                    if (timestamp - last_timestamp > timeout_threshold and
                        (
                            (seq_num is not None and last_seq_num is not None and abs(seq_num - last_seq_num) > seq_diff_threshold) or
                            (proto == 6 and last_flags & 0x01) or  # TCP FIN flag
                            (proto == 6 and last_flags & 0x04)  # TCP RST flag
                        )):  # TCP RST flag
                        flow_key = (src_ip, dst_ip, src_port, dst_port, proto, len(self.flows))  # Create a new flow key
                        self.flows[flow_key] = []

                pkt_flags = int(pkt[scapy.TCP].flags) if scapy.TCP in pkt else 0
                self.flows[flow_key].append((timestamp, proto, pkt_size, pkt_flags, 0, seq_num))  # Direction = 0 for now

        # Sort packets within each flow
        for flow_key in self.flows:
            self.flows[flow_key].sort(key=lambda pkt: pkt[0])  # Sort by timestamp

        if self.verbose:
            print(f"Processed {len(packets)} packets from {self.pcap_file}")
            print(f"Total flows extracted: {len(self.flows)}")

    def compute_features(self):
        data_rows = []
        for flow_key, pkt_list in tqdm(self.flows.items(), desc="Extracting features", unit="flow"):
            src_ip, dst_ip, proto, *_ = flow_key  # Unpack only the first three values from flow_key
            # Sort packets by sequence number if available, otherwise by timestamp
            pkt_list.sort(key=lambda pkt: pkt[5] if pkt[5] is not None else pkt[0])
            flow_duration = pkt_list[-1][0] - pkt_list[0][0]
            pkt_sizes = [pkt[2] for pkt in pkt_list]
            pkt_iats = [pkt_list[i][0] - pkt_list[i - 1][0] for i in range(1, len(pkt_list))]
            max_pkt_size = max(pkt_sizes)
            iat_mean = sum(pkt_iats) / len(pkt_iats) if pkt_iats else 0
            iat_std = (sum((x - iat_mean) ** 2 for x in pkt_iats) / len(pkt_iats)) ** 0.5 if pkt_iats else 0
            row = [src_ip, dst_ip, proto, flow_duration, iat_mean, iat_std, max_pkt_size]
            window = pkt_list[:self.window_size]
            while len(window) < self.window_size:
                window.append((None, None, None, None, None, None))  # Add a placeholder for seq_num
            for j, (timestamp, _, pkt_size, pkt_flags, direction, _) in enumerate(window):  # Ignore seq_num
                prev_timestamp = window[j - 1][0] if j > 0 and window[j - 1][0] is not None else None
                pkt_iat = timestamp - prev_timestamp if timestamp is not None and prev_timestamp is not None else 0
                row.extend([pkt_size, pkt_flags, pkt_iat, direction])
            data_rows.append(row)
        columns = ["Source_IP", "Destination_IP", "Protocol", "Flow_Duration", "Flow_IAT_Mean", "Flow_IAT_Std", "Max_Pkt_Size"]
        for i in range(1, self.window_size + 1):
            columns.extend([f"Pkt_Size{i}", f"Pkt_Flags{i}", f"Pkt_IAT{i}", f"Pkt_Direction{i}"])
        self.df_flows = pd.DataFrame(data_rows, columns=columns)
        numeric_columns = ["Flow_Duration", "Flow_IAT_Mean", "Flow_IAT_Std", "Max_Pkt_Size"]
        self.df_flows[numeric_columns] = self.df_flows[numeric_columns].astype(float)

    def merge_labels(self):
        if not self.labels_file:
            return
        print(f"Loading labels from {self.labels_file}...")
        labels_df = pd.read_csv(self.labels_file)
        labels_df.columns = labels_df.columns.str.strip()

        # Rename columns in labels_df to match self.df_flows
        column_mapping = {
            "Flow Duration": "Flow_Duration",
            "Flow IAT Mean": "Flow_IAT_Mean",
            "Flow IAT Std": "Flow_IAT_Std",
            "Max Packet Length": "Max_Pkt_Size"
        }
        labels_df.rename(columns=column_mapping, inplace=True)

        labels_numeric_columns = ["Flow_Duration", "Flow_IAT_Mean", "Flow_IAT_Std", "Max_Pkt_Size"]
        labels_df[labels_numeric_columns] = labels_df[labels_numeric_columns].astype(float)

        # Normalize numeric columns on copies of the dataframes
        scaler = MinMaxScaler()

        df_flows_normalized = self.df_flows[labels_numeric_columns].copy()
        labels_df_normalized = labels_df[labels_numeric_columns].copy()

        df_flows_normalized = scaler.fit_transform(df_flows_normalized)
        labels_df_normalized = scaler.transform(labels_df_normalized)

        # Sort normalized data for matching
        df_flows_normalized = pd.DataFrame(df_flows_normalized, columns=labels_numeric_columns)
        labels_df_normalized = pd.DataFrame(labels_df_normalized, columns=labels_numeric_columns)

        df_flows_normalized = df_flows_normalized.sort_values(by=labels_numeric_columns)
        labels_df_normalized = labels_df_normalized.sort_values(by=labels_numeric_columns)

        # Perform nearest neighbor search on normalized data
        df_flows_np = df_flows_normalized.to_numpy()
        labels_df_np = labels_df_normalized.to_numpy()

        tree = cKDTree(labels_df_np)
        distances, indices = tree.query(df_flows_np, k=1)

        # Set a distance threshold for defaulting to "Malicious"
        distance_threshold = 0.1  # Adjust this value as needed
        nearest_labels = labels_df.iloc[indices]
        self.df_flows["Label"] = [
            1 if dist > distance_threshold else (0 if label == "BENIGN" else 1)
            for dist, label in zip(distances, nearest_labels["Label"].values)
        ]

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
    if args.labels:
        extractor.merge_labels()
    extractor.save_output()
