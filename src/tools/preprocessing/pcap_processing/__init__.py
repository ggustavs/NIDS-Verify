"""
PCAP Processing Module

Tools for processing network packet captures and extracting machine learning features.

Components:
- extractor: Feature extraction from PCAP files with timestamp disambiguation
- batch_processor: Large PCAP file splitting and batch processing
"""

from .extractor import FeatureExtractor
from .batch_processor import BatchPcapProcessor

__all__ = ['FeatureExtractor', 'BatchPcapProcessor']
