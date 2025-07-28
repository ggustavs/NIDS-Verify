"""
NIDS-Verify Preprocessing CLI

Unified command-line interface for NIDS-Verify data preprocessing tools.
Provides easy access to feature extraction, visualization, and data processing.
"""

import argparse
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_features(args):
    """Run feature extraction."""
    try:
        from src.tools.preprocessing.pcap_processing.extractor import main as extractor_main
        
        # Convert args to format expected by extractor
        sys.argv = [
            'extractor.py',
            args.pcap_file,
            '--window', str(args.window)
        ]
        
        if args.labels:
            sys.argv.extend(['--labels', args.labels])
        if args.output:
            sys.argv.extend(['--output', args.output])
        if args.quiet:
            sys.argv.append('--quiet')
            
        return extractor_main()
        
    except ImportError as e:
        logger.error(f"Failed to import feature extractor: {e}")
        return 1
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return 1


def batch_process(args):
    """Run batch processing."""
    try:
        from src.tools.preprocessing.pcap_processing.batch_processor import main as batch_main
        
        # Convert args to format expected by batch processor
        sys.argv = [
            'batch_processor.py',
            args.input_pcap,
            args.output_csv,
            '--labels', args.labels
        ]
        
        if args.output_dir:
            sys.argv.extend(['--output-dir', args.output_dir])
        if args.window:
            sys.argv.extend(['--window', str(args.window)])
        if args.size_limit:
            sys.argv.extend(['--size-limit', args.size_limit])
        if args.no_cleanup:
            sys.argv.append('--no-cleanup')
        if args.quiet:
            sys.argv.append('--quiet')
            
        return batch_main()
        
    except ImportError as e:
        logger.error(f"Failed to import batch processor: {e}")
        return 1
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 1


def visualize(args):
    """Run visualization."""
    try:
        from src.tools.preprocessing.visualization.plotter import main as viz_main
        
        # Convert args to format expected by visualizer
        sys.argv = ['plotter.py', args.data_file]
        
        if args.output_dir:
            sys.argv.extend(['--output-dir', args.output_dir])
        if args.report:
            sys.argv.append('--report')
        if args.histogram:
            sys.argv.extend(['--histogram', args.histogram])
        if args.scatter:
            sys.argv.extend(['--scatter'] + args.scatter)
        if args.correlation:
            sys.argv.append('--correlation')
        if args.importance:
            sys.argv.append('--importance')
        if args.sample_size:
            sys.argv.extend(['--sample-size', str(args.sample_size)])
            
        return viz_main()
        
    except ImportError as e:
        logger.error(f"Failed to import visualizer: {e}")
        return 1
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NIDS-Verify Preprocessing - Data processing tools for network intrusion detection research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from PCAP
  nids-preprocess extract sample.pcap --labels flows.csv --window 10
  
  # Process large PCAP in batches
  nids-preprocess batch large.pcap output.csv --labels flows.csv --size-limit 1000m
  
  # Generate comprehensive visualizations
  nids-preprocess visualize features.csv --report
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract features from PCAP files')
    extract_parser.add_argument('pcap_file', help='Path to PCAP file')
    extract_parser.add_argument('--labels', help='Path to labels CSV file')
    extract_parser.add_argument('--output', help='Output CSV file path')
    extract_parser.add_argument('--window', type=int, default=10, help='Feature window size')
    extract_parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process large PCAP files in batches')
    batch_parser.add_argument('input_pcap', help='Input PCAP file')
    batch_parser.add_argument('output_csv', help='Output CSV file')
    batch_parser.add_argument('--labels', required=True, help='Labels CSV file')
    batch_parser.add_argument('--output-dir', default='splits', help='Temporary directory')
    batch_parser.add_argument('--window', type=int, default=10, help='Feature window size')
    batch_parser.add_argument('--size-limit', default='1000m', help='Split size limit')
    batch_parser.add_argument('--no-cleanup', action='store_true', help='Keep temporary files')
    batch_parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate data visualizations')
    viz_parser.add_argument('data_file', help='CSV file with features')
    viz_parser.add_argument('--output-dir', default='visualizations', help='Output directory')
    viz_parser.add_argument('--report', action='store_true', help='Generate full report')
    viz_parser.add_argument('--histogram', help='Feature for histogram')
    viz_parser.add_argument('--scatter', nargs=2, help='Two features for scatter plot')
    viz_parser.add_argument('--correlation', action='store_true', help='Correlation matrix')
    viz_parser.add_argument('--importance', action='store_true', help='Feature importance')
    viz_parser.add_argument('--sample-size', type=int, default=5000, help='Sample size for plots')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command
    if args.command == 'extract':
        return extract_features(args)
    elif args.command == 'batch':
        return batch_process(args)
    elif args.command == 'visualize':
        return visualize(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit(main())
