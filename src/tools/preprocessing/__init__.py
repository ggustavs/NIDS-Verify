"""
NIDS-Verify Tools Package

This package contains utilities for:
- PCAP processing and feature extraction
- Data visualization and analysis
- Vehicle-lang verification tools
- Hyperrectangle generation for adversarial training

Organization:
- pcap_processing/: PCAP file processing and feature extraction
- visualization/: Data analysis and plotting utilities
- verification/: Vehicle-lang integration and formal verification tools
"""

__version__ = "1.0.0"
__author__ = "NIDS-Verify Research Project"

# Import main tool classes for convenience
try:
    from .pcap_processing.extractor import FeatureExtractor
    from .visualization.plotter import DataVisualizer

    __all__ = ["FeatureExtractor", "DataVisualizer"]
except ImportError as e:
    # Handle missing dependencies gracefully
    import logging

    logging.warning(f"Some tools may not be available due to missing dependencies: {e}")
    __all__ = []
