"""Projected Gradient Descent on research-defined hyperrectangles (Flood et al. 2025).

Each pattern (hulk, goodHTTP1, ...) is a hyperrectangle in feature space. PGD
clamps adversarial examples to that absolute box on each step.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.nn.functional import cross_entropy

# --- per-feature-class bound constants ------------------------------------
# Bounds are in normalised [0,1] feature space (matching the Vehicle spec).

_PROTOCOL_TCP = (0.0, 0.0)

_HANDSHAKE_DIRS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (1.0, 1.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (1.0, 1.0),
]
_FREE_DIR = (0.0, 1.0)
_FREE_FLAG = (0.0, 1.0)
_FREE_IAT = (0.0, 1.0)
_FREE_SIZE = (0.0, 1.0)

_HANDSHAKE_FLAGS: list[tuple[float, float]] = [
    (2 / 256, 2 / 256),  # SYN
    (18 / 256, 18 / 256),  # SYN+ACK
    (16 / 256, 16 / 256),  # ACK
    (24 / 256, 24 / 256),  # PSH+ACK (HTTP request)
    (16 / 256, 16 / 256),  # ACK (HTTP response)
]

_GOOD_IAT = (0.000001, 0.05)

_HANDSHAKE_SIZES: list[tuple[float, float]] = [
    (52 / 1000, 52 / 1000),  # SYN: 52B
    (52 / 1000, 52 / 1000),  # SYN+ACK: 52B
    (40 / 1000, 40 / 1000),  # ACK: 40B
    (100 / 1000, 500 / 1000),  # HTTP request: 100-500B
    (40 / 1000, 40 / 1000),  # HTTP response ACK: 40B
]


@dataclass(frozen=True)
class HyperrectPattern:
    """Per-feature [lower, upper] bounds for a labelled traffic pattern."""

    bounds: np.ndarray  # shape (F, 2)

    def to_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        return torch.as_tensor(self.bounds, dtype=torch.float32, device=device)


def _build_pattern(
    time_elapsed: tuple[float, float],
    flag_bounds: list[tuple[float, float]],
    iat_bound: tuple[float, float],
    size_bounds: list[tuple[float, float]],
    pkts_length: int = 10,
) -> np.ndarray:
    """Construct a (2 + 4*pkts_length, 2) bounds array.

    Layout: [time_elapsed, protocol, *directions, *flags, *iats, *sizes]
    where direction/flag/iat/size groups have `pkts_length` entries.
    """
    bounds: list[tuple[float, float]] = [time_elapsed, _PROTOCOL_TCP]

    dirs = list(_HANDSHAKE_DIRS) + [_FREE_DIR] * (pkts_length - len(_HANDSHAKE_DIRS))
    bounds.extend(dirs[:pkts_length])

    flags = list(flag_bounds) + [_FREE_FLAG] * (pkts_length - len(flag_bounds))
    bounds.extend(flags[:pkts_length])

    iats: list[tuple[float, float]] = [_FREE_IAT]
    iats += [iat_bound] * (pkts_length - 1)
    bounds.extend(iats[:pkts_length])

    sizes = list(size_bounds) + [_FREE_SIZE] * (pkts_length - len(size_bounds))
    bounds.extend(sizes[:pkts_length])

    return np.asarray(bounds, dtype=np.float32)


def _build_patterns(pkts_length: int = 10) -> dict[str, HyperrectPattern]:
    flags_handshake = _HANDSHAKE_FLAGS
    sizes_handshake = _HANDSHAKE_SIZES
    free_iats = _FREE_IAT

    return {
        "goodHTTP1": HyperrectPattern(
            _build_pattern(
                (0.0, 0.0),
                flags_handshake,
                _GOOD_IAT,
                sizes_handshake,
                pkts_length,
            )
        ),
        "goodHTTP2": HyperrectPattern(
            _build_pattern(
                (0.002, 1.0),
                flags_handshake,
                _GOOD_IAT,
                sizes_handshake,
                pkts_length,
            )
        ),
        "hulk": HyperrectPattern(
            _build_pattern(
                (1e-14, 1e-3),
                flags_handshake,
                free_iats,
                sizes_handshake,
                pkts_length,
            )
        ),
        "slowIATsAttacks": HyperrectPattern(
            _build_pattern(
                (1e-14, 1e-3),
                flags_handshake,
                free_iats,
                sizes_handshake,
                pkts_length,
            )
        ),
        "invalid": HyperrectPattern(
            _build_pattern(
                (0.0, 1.0),
                [_FREE_FLAG] * pkts_length,
                _FREE_IAT,
                [_FREE_SIZE] * pkts_length,
                pkts_length,
            )
        ),
    }


_PATTERNS = _build_patterns()


def get_research_hyperrectangles() -> dict[str, np.ndarray]:
    """Return per-pattern (F, 2) bounds arrays."""
    return {name: p.bounds.copy() for name, p in _PATTERNS.items()}


def _bounds_tensor(pattern: str, num_features: int, device: torch.device) -> torch.Tensor:
    if pattern not in _PATTERNS:
        logger.warning(f"Unknown pattern '{pattern}', falling back to 'hulk'")
        pattern = "hulk"
    bounds = _PATTERNS[pattern].to_tensor(device=device)
    if bounds.shape[0] < num_features:
        # pad with [0,1] for any extra features
        pad = torch.tensor(
            [[0.0, 1.0]] * (num_features - bounds.shape[0]),
            dtype=torch.float32,
            device=device,
        )
        bounds = torch.cat([bounds, pad], dim=0)
    return bounds[:num_features]


def project_to_hyperrectangle(x: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Clamp x to [bounds[:,0], bounds[:,1]] feature-wise."""
    return torch.clamp(x, min=bounds[:, 0], max=bounds[:, 1])


def pgd_attack_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    bounds: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """One PGD step: x_{t+1} = Π(x_t + α·sign(∇L))."""
    x_prev = x.detach().requires_grad_(True)
    logits = model(x_prev)
    loss = cross_entropy(logits, y.long())
    grad = torch.autograd.grad(loss, x_prev)[0]
    x_next = x_prev.detach() + step_size * torch.sign(grad)
    return project_to_hyperrectangle(x_next, bounds)


def generate_pgd_adversarial_examples(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    attack_rects: list[np.ndarray] | None = None,
    epsilon: float = 0.1,
    num_steps: int = 3,
    step_size: float = 0.01,
    attack_pattern: str = "hulk",
) -> torch.Tensor:
    """Constraint-guided PGD: clamp to absolute hyperrectangle each step."""
    num_features = x.shape[1]
    if attack_rects is not None:
        bounds = torch.as_tensor(np.stack(attack_rects), dtype=torch.float32, device=x.device)
    else:
        bounds = _bounds_tensor(attack_pattern, num_features, x.device)

    x_adv = x + torch.empty_like(x).uniform_(-epsilon / 10, epsilon / 10)
    x_adv = project_to_hyperrectangle(x_adv, bounds)

    for _ in range(num_steps):
        x_adv = pgd_attack_step(model, x_adv, y, bounds, step_size)
    return x_adv


def create_attack_rectangles(
    attack_pattern: str = "hulk",
    input_size: int = 42,
    feature_names: list[str] | None = None,  # accepted for backward compatibility
) -> list[np.ndarray]:
    """Return per-feature [min, max] bounds as a list of length-2 arrays.

    Kept as a list-of-arrays for compatibility with callers that expect that shape.
    """
    if attack_pattern == "mixed":
        attack_pattern = "hulk"
    bounds = _bounds_tensor(attack_pattern, input_size, torch.device("cpu")).numpy()
    return [bounds[i].astype(np.float32) for i in range(input_size)]
