# NIDS-Verify Data Processing Tools

A comprehensive suite of modular data processing tools for network intrusion detection research, now unified under `src/tools/nids_tools/`.

## Overview

This directory contains the restructured and enhanced data processing tools:

```
src/preprocessing/nids_tools/
├── __init__.py                 # Package entry point
├── pcap_processing/            # Network packet processing
│   ├── __init__.py
│   ├── extractor.py           # Feature extraction from PCAP files
│   └── batch_processor.py     # Large file batch processing
├── visualization/              # Data analysis and plotting
│   ├── __init__.py
│   └── plotter.py             # Comprehensive visualization tools
└── legacy/                     # Archived original flat scripts
    ├── extract_features.py
    ├── split_and_extract.py
    └── visualize.py
```

## Installation

From the repository root (using uv for dependency management):

```bash
# Install all dependencies with uv
uv sync

# Or add specific dependencies
uv add pandas numpy matplotlib seaborn scikit-learn scapy tqdm

# Run tools with uv
uv run python ./tools/cli.py --help
```

## Usage

### Unified CLI System

All NIDS-Verify tools use a consistent CLI approach through `pyproject.toml` entry points:

- **`nids-train`**: Model training and management
- **`nids-models`**: Model registry operations  
- **`nids-preprocess`**: Data preprocessing and analysis tools

### Command Line Interface

The tools are accessible through the unified `nids-preprocess` CLI entry point (alongside `nids-train` and `nids-models`):

```bash
# Extract features from PCAP
uv run nids-preprocess extract sample.pcap --labels flows.csv --window 10

# Process large PCAP in batches
uv run nids-preprocess batch large.pcap output.csv --labels flows.csv --size-limit 1000m

# Generate comprehensive visualizations
uv run nids-preprocess visualize features.csv --report
```

### Python API

Import and use tools programmatically:

```python
from src.preprocessing.nids_tools import FeatureExtractor, DataVisualizer

# Feature extraction
extractor = FeatureExtractor(window_size=10)
features = extractor.extract_from_pcap('sample.pcap', 'labels.csv')

# Visualization
visualizer = DataVisualizer('features.csv')
visualizer.generate_report('output_dir/')
```

## Features

### PCAP Processing
- **Feature Extraction**: Extract 78+ network flow features with timestamp disambiguation
- **Batch Processing**: Handle large PCAP files by splitting into manageable chunks
- **Label Integration**: Merge extracted features with ground truth labels
- **Memory Optimization**: Efficient processing of multi-gigabyte files

### Visualization
- **Correlation Analysis**: Feature correlation matrices with statistical significance
- **Distribution Plots**: Histograms, scatter plots, and probability density functions
- **Feature Importance**: Random Forest and permutation-based importance scoring
- **Attack Analysis**: Class-specific visualizations and separation metrics
- **Report Generation**: Comprehensive HTML/PDF reports with all visualizations

## Migration from Legacy Tools

The legacy flat scripts have been moved to `tools/legacy/` and enhanced:

| Legacy Script | New Module | Enhancements |
|---------------|------------|--------------|
| `extract_features.py` | `pcap_processing.extractor` | Better timestamp handling, memory optimization, CLI improvements |
| `split_and_extract.py` | `pcap_processing.batch_processor` | Configurable split sizes, cleanup options, progress tracking |
| `visualize.py` | `visualization.plotter` | More plot types, statistical analysis, report generation |

## Dependencies

- **Core**: pandas, numpy, matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Network Processing**: scapy
- **Optional**: jupyter (for notebook integration)

## Architecture

The modular design provides:

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Reusability**: Components can be imported and used independently
3. **Extensibility**: Easy to add new tools or modify existing ones
4. **Testing**: Each module can be unit tested in isolation
5. **Documentation**: Clear API documentation and examples

## Contributing

When adding new tools:

1. Place them in the appropriate subdirectory
2. Update the relevant `__init__.py` file
3. Add CLI integration in `cli.py`
4. Include comprehensive docstrings
5. Add usage examples to this README

## Examples

See the CLI help for detailed usage examples:

```bash
uv run nids-preprocess --help
uv run nids-preprocess extract --help
uv run nids-preprocess visualize --help
```

All CLI commands are available alongside the other repository tools:

```bash
uv run nids-train --help       # Model training
uv run nids-models --help      # Model management  
uv run nids-preprocess --help  # Data preprocessing tools
```

Or check the Jupyter notebook `nb/data_analysis.ipynb` for interactive examples.
