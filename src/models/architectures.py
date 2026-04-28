"""MLP architectures for NIDS classification."""

from torch import nn

ARCHITECTURES: dict[str, list[int]] = {
    "small":   [128],
    "mid":     [256, 128],
    "mid2":    [512, 256, 128],
    "mid3":    [512, 256, 256, 128],
    "mid4":    [512, 512, 256, 128],
    "big":     [1024, 512, 256, 128],
    "big2":    [1024, 512, 512, 256],
    "big3":    [1024, 1024, 512, 256],
    "big4":    [2048, 1024, 512, 256],
    "massive": [2048, 1024, 512, 256, 128],
}

MODEL_TYPES = list(ARCHITECTURES.keys())


def make_mlp(input_size: int, hidden: list[int], output_size: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_size
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, output_size))
    return nn.Sequential(*layers)


def create_model(input_size: int, model_type: str) -> nn.Sequential:
    if model_type not in ARCHITECTURES:
        raise ValueError(f"Unknown model type: {model_type}. Available: {MODEL_TYPES}")
    return make_mlp(input_size, ARCHITECTURES[model_type])
