# NIDS-Verify

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![UV Package Manager](https://img.shields.io/badge/UV-package%20manager-green.svg)](https://github.com/astral-sh/uv)

A Network Intrusion Detection System (NIDS) that combines adversarial training with formal verification using Vehicle-lang. This repository explores verifiable global constraints and adversarial training to improve NIDS model robustness and generalization.

## Features

- **Vehicle-lang Integration**: Formal verification specifications for attack patterns
- **Adversarial Training**: PGD-based training with hyperrectangle constraints
- **Experiment Tracking**: MLflow integration for model management and monitoring
- **Unified Training**: Single interface for base and adversarial training

## Getting Started

### Prerequisites

- Python 3.12+
- [UV Package Manager](https://github.com/astral-sh/uv)
- [Vehicle-lang](https://github.com/vehicle-lang/vehicle) (optional, for formal verification)

### Installation

```bash
git clone https://github.com/yourusername/NIDS-Verify-Personal.git
cd NIDS-Verify-Personal
uv sync
```

### Usage

Train a base model:
```bash
uv run nids-train --model-type small --training-type base --epochs 10
```

Train an adversarial model with research patterns:
```bash
uv run nids-train --model-type small --training-type adversarial --epochs 15 --attack-pattern hulk
```

Register pre-trained models:
```bash
uv run nids-models register --dir models/tf/DoS2
```

Manage models:
```bash
# List registered models
uv run nids-models list

# Evaluate a model
uv run nids-models evaluate nids-dos2-big4-eps0.3-bs4096

# Compare all models
uv run nids-models compare

# Find best model
uv run nids-models best --metric final_val_accuracy

# Load a model for testing
uv run nids-models load nids_small_adversarial --version 2
```

## Project Structure

```
NIDS-Verify-Personal/
├── src/                              # Core source code
│   ├── config.py                     # Centralized configuration
│   ├── main.py                       # CLI entry point
│   ├── attacks/                      # Adversarial attack implementations
│   │   └── pgd.py                   # PGD with hyperrectangle constraints
│   ├── data/                         # Data loading utilities
│   ├── models/                       # Model architectures & management
│   │   ├── architectures.py         # Neural network definitions
│   │   ├── registry.py              # MLflow Model Registry
│   │   └── manage.py                # Model management CLI
│   ├── tools/                        # Vehicle-lang integration
│   │   └── vehicle_hyperrectangles.py
│   ├── training/                     # Training pipelines
│   │   └── trainer.py               # Unified NIDSTrainer
│   └── utils/                        # Logging & performance utilities
├── vehicle/                          # Vehicle-lang specifications
│   ├── global.vcl                   # Global definitions
│   └── nids_hyperrectangle_spec.vcl # Attack pattern specifications
├── nb/                              # Jupyter notebooks
│   ├── data_analysis.ipynb         # Exploratory data analysis
│   └── vehicle_hyperrectangle_generator.ipynb  # Interactive tool
├── mlruns/                          # MLflow experiment tracking
├── data/                            # Datasets (CIC-IDS-2018, DetGen, etc.)
└── logs/                            # Training and gradient logs
```

## Model Architectures

Available model types:
- **small**: ~1K parameters (quick experiments)
- **mid**: ~10K parameters (balanced performance)
- **big**: ~100K parameters (high capacity)
- **massive**: ~1M parameters (maximum performance)

## Experiment Tracking

The project uses MLflow for experiment tracking and model management:

```bash
mlflow ui  # View experiments at http://localhost:5000
```

Tracked metrics include:
- Training/validation accuracy and loss
- Gradient norms and health indicators
- Model parameters and hyperparameters
- Training time and performance metrics

Gradient health monitoring provides real-time feedback:
- Green: Normal gradient flow
- Yellow: Potential gradient issues
- Red: Gradient problems detected

## Formal Verification

The system defines formal specifications for attack patterns using Vehicle-lang:

```vehicle
// Example: DoS attack pattern
property dos_attack_robustness(input: Vector, output: Vector) -> Bool = {
    forall attack_region in dos_hyperrectangles.
        input in attack_region =>
        model_prediction(input) == ATTACK_LABEL
}
```

Verification properties include:
- **Robustness**: Model maintains predictions within attack regions
- **Consistency**: Similar inputs produce similar outputs
- **Completeness**: Coverage of attack space

## Adversarial Training

The PGD implementation projects adversarial examples to valid hyperrectangles:

```python
trainer = NIDSTrainer(model, config)
history = trainer.train(
    data, val_data,
    training_type="adversarial",
    attack_bounds=perturbation_bounds
)
```

Attack boundaries are generated using:
- Statistical analysis (percentiles, IQR, mean±std)
- Expert knowledge integration
- Data-driven suggestions

## Datasets

Supported datasets:
- **CIC-IDS-2018**: Comprehensive intrusion detection dataset
- **DetGen**: SSH-specific attack data
- **Custom**: DoS-focused preprocessed datasets

Data preprocessing includes:
- Feature normalization and scaling
- Label encoding for multi-class classification
- Stratified train/validation/test splitting

## Configuration

Centralized configuration in `src/config.py`:

```python
# Model configuration
model.model_type = "small"
model.dropout_rate = 0.3

# Training configuration
training.learning_rate = 0.001
training.batch_size = 32
training.epochs = 50

# Attack configuration
attack.epsilon = 0.1
attack.alpha = 0.01
attack.num_steps = 10
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
