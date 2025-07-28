"""
NIDS-Verify Preprocessing Collection

This module contains various preprocessing tools for the NIDS-Verify project:

- model_manager.py: Legacy model management utilities
- vehicle_hyperrectangles.py: Legacy Vehicle-lang integration
- nids_tools/: Modular data processing and analysis tools
  - pcap_processing/: Network packet processing
  - visualization/: Data analysis and plotting
"""

# Import legacy tools for backward compatibility
# Import modular tools for convenience
try:
    from .preprocessing import FeatureExtractor, DataVisualizer
    __all__ = ['FeatureExtractor', 'DataVisualizer']
except ImportError:
    __all__ = []
