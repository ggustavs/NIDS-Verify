"""
PCAP Processing Module

Tools for processing network packet captures and extracting machine learning features.

Components:
- extractor: Feature extraction from PCAP files with timestamp disambiguation
- batch_processor: Large PCAP file splitting and batch processing
"""

from .batch_processor import BatchPcapProcessor
from .extractor import FeatureExtractor

__all__ = ["FeatureExtractor", "BatchPcapProcessor"]
