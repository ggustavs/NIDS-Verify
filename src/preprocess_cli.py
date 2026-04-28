"""NIDS-Verify preprocessing CLI: PCAP feature extraction and EDA visualisations."""

import argparse
import sys

from loguru import logger


def extract_features(args: argparse.Namespace) -> int:
    from src.pcap.extractor import FeatureExtractor

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
        extractor.get_split_flow_report().to_csv(args.split_report, index=False)
        logger.info(f"Split flow report: {args.split_report}")

    logger.info(
        f"Extraction done: complete={len(extractor.complete_flows)} "
        f"split={len(extractor.split_flows)} → {output_file}"
    )
    return 0


def batch_process(args: argparse.Namespace) -> int:
    from src.pcap.batch import BatchPcapProcessor

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


def visualize(args: argparse.Namespace) -> int:
    from src.viz import DataVisualizer

    viz = DataVisualizer(args.data_file, args.output_dir)
    viz.load_data()

    if args.report:
        viz.generate_comprehensive_report()
        return 0

    plots = 0
    if args.histogram:
        viz.plot_feature_histogram(args.histogram)
        plots += 1
    if args.scatter:
        viz.plot_feature_scatter(args.scatter[0], args.scatter[1], args.sample_size)
        plots += 1
    if args.correlation:
        viz.plot_correlation_matrix()
        plots += 1
    if args.importance:
        viz.plot_label_correlations()
        plots += 1

    if plots == 0:
        logger.warning("No plot requested. Use --report for full analysis.")
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NIDS-Verify preprocessing tools (PCAP extraction, batch processing, EDA)."
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    extract_p = sub.add_parser("extract", help="Extract features from a PCAP")
    extract_p.add_argument("pcap_file")
    extract_p.add_argument("--labels", required=True)
    extract_p.add_argument("--output")
    extract_p.add_argument("--split-report")
    extract_p.add_argument("--window", type=int, default=10)
    extract_p.add_argument("--quiet", action="store_true")

    batch_p = sub.add_parser("batch", help="Process a large PCAP in chunks")
    batch_p.add_argument("input_pcap")
    batch_p.add_argument("output_csv")
    batch_p.add_argument("--labels", required=True)
    batch_p.add_argument("--output-dir", default="splits")
    batch_p.add_argument("--window", type=int, default=10)
    batch_p.add_argument("--size-limit", default="2000m")
    batch_p.add_argument("--no-cleanup", action="store_true")
    batch_p.add_argument("--quiet", action="store_true")

    viz_p = sub.add_parser("visualize", help="Generate EDA plots")
    viz_p.add_argument("data_file")
    viz_p.add_argument("--output-dir", default="visualizations")
    viz_p.add_argument("--report", action="store_true")
    viz_p.add_argument("--histogram")
    viz_p.add_argument("--scatter", nargs=2, metavar=("X", "Y"))
    viz_p.add_argument("--correlation", action="store_true")
    viz_p.add_argument("--importance", action="store_true")
    viz_p.add_argument("--sample-size", type=int, default=5000)

    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not args.command:
        _build_parser().print_help()
        return 1

    handlers = {"extract": extract_features, "batch": batch_process, "visualize": visualize}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
