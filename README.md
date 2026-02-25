# NIDS-Verify

Network Intrusion Detection System with adversarial training and formal verification.

## Features

- **PCAP Processing**: Extract features from packet captures
- **Adversarial Training**: PGD-based robust model training
- **Formal Verification**: Vehicle-lang constraint verification
- **MLflow Tracking**: Experiment and model management

## Installation

```bash
git clone <repository-url>
cd NIDS-Verify
uv sync
```

Requires Python 3.12+ and [UV package manager](https://github.com/astral-sh/uv).

## Quick Start

### Preprocess PCAPs
```bash
# Extract features
nids-preprocess extract input.pcap --labels flows.csv --output features.csv

# Process large files (auto-splits)
nids-preprocess batch large.pcap output.csv --labels flows.csv
```

### Train Models
```bash
# Base training
nids-train --model-type small --training-type base --epochs 10

# Adversarial training
nids-train --model-type small --training-type adversarial --epochs 10
```

### Evaluate Models
```bash
# Empirical + adversarial evaluation
nids-evaluate --model nids-dos2-small-base --dataset data/test.csv --empirical --security

# With formal verification (requires Marabou)
nids-evaluate --model nids-dos2-small-base --dataset data/test.csv --formal --empirical
```

## Documentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed structure and CLI reference.

## Datasets

Tested on:
- CIC-IDS-2017: Network intrusion detection dataset
- DoS-specific preprocessed subsets

## Model Registry

Models tracked with MLflow:
```bash
mlflow ui  # Browse experiments at http://localhost:5000
```

## License

See LICENSE file.
