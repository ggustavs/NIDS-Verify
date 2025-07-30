"""
PGD (Projected Gradient Descent) attack implementation
"""

import numpy as np
import tensorflow as tf

from src.utils.logging import get_logger

logger = get_logger(__name__)


def project_to_hyperrectangle(
    x: tf.Tensor, x_orig: tf.Tensor, attack_rects: list[np.ndarray], absolute_bounds: bool = False
) -> tf.Tensor:
    """
    Project adversarial examples to hyperrectangle constraints

    Args:
        x: Current adversarial examples
        x_orig: Original clean examples
        attack_rects: List of hyperrectangle constraints for each feature
        absolute_bounds: If True, treat attack_rects as absolute bounds;
                        if False, treat as perturbation bounds relative to x_orig

    Returns:
        Projected adversarial examples
    """
    x_proj = tf.identity(x)

    for i, rect in enumerate(attack_rects):
        if len(rect) >= 2:
            if absolute_bounds:
                # Use absolute bounds from research hyperrectangles
                # Handle both tensor and numpy/python cases
                if isinstance(rect[0], tf.Tensor):
                    lower_bound = rect[0]
                    upper_bound = rect[1]
                else:
                    lower_bound = tf.constant(float(rect[0]), dtype=tf.float32)
                    upper_bound = tf.constant(float(rect[1]), dtype=tf.float32)
            else:
                # Calculate bounds relative to original input (perturbation mode)
                if isinstance(rect[0], tf.Tensor):
                    lower_bound = x_orig[:, i] + rect[0]
                    upper_bound = x_orig[:, i] + rect[1]
                else:
                    lower_bound = x_orig[:, i] + tf.constant(float(rect[0]), dtype=tf.float32)
                    upper_bound = x_orig[:, i] + tf.constant(float(rect[1]), dtype=tf.float32)

            # Clip to bounds
            if i == 0:
                # First column
                x_proj = tf.concat(
                    [
                        tf.expand_dims(tf.clip_by_value(x_proj[:, i], lower_bound, upper_bound), 1),
                        x_proj[:, 1:],
                    ],
                    axis=1,
                )
            else:
                # Other columns
                x_proj = tf.concat(
                    [
                        x_proj[:, :i],
                        tf.expand_dims(tf.clip_by_value(x_proj[:, i], lower_bound, upper_bound), 1),
                        x_proj[:, i + 1 :],
                    ],
                    axis=1,
                )

    return x_proj


def project_to_research_hyperrectangle(x: tf.Tensor, hyperrect_pattern: str = "hulk") -> tf.Tensor:
    """
    Project adversarial examples to research-based hyperrectangle constraints.
    This uses the absolute bounds defined in the original research.

    Args:
        x: Current adversarial examples
        hyperrect_pattern: Which research hyperrectangle pattern to use

    Returns:
        Projected adversarial examples within research-defined bounds
    """
    hyperrects = get_research_hyperrectangles()

    if hyperrect_pattern not in hyperrects:
        logger.warning(f"Unknown hyperrectangle pattern '{hyperrect_pattern}', using 'hulk'")
        hyperrect_pattern = "hulk"

    bounds = hyperrects[hyperrect_pattern]

    # Convert bounds to tensors for vectorized operations
    lower_bounds = []
    upper_bounds = []

    # Determine how many features we can bound (limited by both bounds and input)
    input_features = tf.shape(x)[1]
    max_features = min(len(bounds), x.shape[1] if x.shape[1] is not None else len(bounds))

    # Extract bounds up to max_features
    for i in range(max_features):
        if i < len(bounds) and isinstance(bounds[i], list) and len(bounds[i]) >= 2:
            lower_bounds.append(bounds[i][0])
            upper_bounds.append(bounds[i][1])
        else:
            # If no bounds for this feature, use very wide bounds (essentially unbounded)
            lower_bounds.append(-1e6)
            upper_bounds.append(1e6)

    if not lower_bounds:
        return x  # No bounds to apply

    # Convert to tensors
    lower_bounds_tensor = tf.constant(lower_bounds, dtype=tf.float32)
    upper_bounds_tensor = tf.constant(upper_bounds, dtype=tf.float32)

    # Extract the features that have bounds
    x_bounded = x[:, : len(lower_bounds)]

    # Apply clipping
    x_clipped = tf.clip_by_value(x_bounded, lower_bounds_tensor, upper_bounds_tensor)

    # Reconstruct the full tensor
    if len(lower_bounds) < input_features:
        # Concatenate clipped bounded features with unbounded features
        x_proj = tf.concat([x_clipped, x[:, len(lower_bounds) :]], axis=1)
    else:
        x_proj = x_clipped

    return x_proj


@tf.function
def pgd_attack_step(
    model: tf.keras.Model,
    x: tf.Tensor,
    y: tf.Tensor,
    attack_rects: list[np.ndarray],
    step_size: float,
    use_research_bounds: bool = True,
) -> tf.Tensor:
    """
    Single PGD attack step

    Args:
        model: Target model
        x: Current adversarial examples
        y: True labels
        attack_rects: Hyperrectangle constraints
        step_size: Step size for gradient ascent
        use_research_bounds: Whether to use research-based absolute bounds

    Returns:
        Updated adversarial examples
    """
    x_orig = tf.identity(x)

    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = model(x, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        loss = tf.reduce_mean(loss)

    # Calculate gradients
    gradients = tape.gradient(loss, x)

    # Gradient ascent step (we want to maximize loss)
    x_adv = x + step_size * tf.sign(gradients)

    # Project to constraints using appropriate method
    if use_research_bounds:
        # Use absolute bounds from research hyperrectangles
        x_adv = project_to_hyperrectangle(x_adv, x_orig, attack_rects, absolute_bounds=True)
    else:
        # Use perturbation bounds
        x_adv = project_to_hyperrectangle(x_adv, x_orig, attack_rects, absolute_bounds=False)

    return x_adv


def generate_pgd_adversarial_examples(
    model: tf.keras.Model,
    x: tf.Tensor,
    y: tf.Tensor,
    attack_rects: list[np.ndarray] = None,
    epsilon: float = 0.1,
    num_steps: int = 3,
    step_size: float = 0.01,
    attack_pattern: str = "hulk",
) -> tf.Tensor:
    """
    Generate adversarial examples using PGD attack with research-based hyperrectangles

    Args:
        model: Target model
        x: Clean input examples
        y: True labels
        attack_rects: Hyperrectangle constraints (if None, will use research patterns)
        epsilon: Maximum perturbation magnitude (for initialization)
        num_steps: Number of PGD steps
        step_size: Step size for each iteration
        attack_pattern: Which research attack pattern to use

    Returns:
        Adversarial examples
    """
    # Use research-based attack rectangles if none provided
    if attack_rects is None:
        input_size = tf.shape(x)[1].numpy() if hasattr(tf.shape(x)[1], "numpy") else 42
        attack_rects = create_attack_rectangles(
            attack_pattern=attack_pattern, input_size=input_size
        )
        logger.info(f"Using research-based '{attack_pattern}' hyperrectangles for PGD attack")

    # Initialize adversarial examples with small random noise
    x_adv = x + tf.random.uniform(tf.shape(x), -epsilon / 10, epsilon / 10)

    # Project to research hyperrectangle immediately
    x_adv = project_to_research_hyperrectangle(x_adv, attack_pattern)

    # PGD iterations with research-based constraints
    for _ in range(num_steps):
        x_adv = pgd_attack_step(model, x_adv, y, attack_rects, step_size, use_research_bounds=True)

    return x_adv


def get_research_hyperrectangles():
    """
    Get the research-based hyperrectangle definitions from the original paper.
    These hyperrectangles represent specific attack patterns and benign traffic.

    Returns:
        Dictionary containing different attack pattern hyperrectangles
    """
    # Benign HTTP traffic with time elapsed = 0.0
    goodHTTP1 = [  # noqa: N806
        [0.0, 0.0],
        [0.0, 0.0],  # Time elapsed, protocol
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt direction
        [2 / 256, 2 / 256],
        [18 / 256, 18 / 256],
        [16 / 256, 16 / 256],
        [24 / 256, 24 / 256],
        [16 / 256, 16 / 256],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt flags
        [0.0, 1.0],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],  # Pkt IATs
        [52 / 1000, 52 / 1000],
        [52 / 1000, 52 / 1000],
        [40 / 1000, 40 / 1000],
        [100 / 1000, 500 / 1000],
        [40 / 1000, 40 / 1000],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]  # Pkt size

    # Benign HTTP traffic with time elapsed between [0.001, 1.0]
    goodHTTP2 = [  # noqa: N806
        [0.002, 1.0],
        [0.0, 0.0],  # Time elapsed, protocol
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt direction
        [2 / 256, 2 / 256],
        [18 / 256, 18 / 256],
        [16 / 256, 16 / 256],
        [24 / 256, 24 / 256],
        [16 / 256, 16 / 256],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt flags
        [0.0, 1.0],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],
        [0.000001, 0.05],  # Pkt IATs
        [52 / 1000, 52 / 1000],
        [52 / 1000, 52 / 1000],
        [40 / 1000, 40 / 1000],
        [100 / 1000, 500 / 1000],
        [40 / 1000, 40 / 1000],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]  # Pkt size

    # HULK DoS attack pattern
    hulk = [
        [0.00000000000001, 0.001],
        [0.0, 0.0],  # Time elapsed, protocol
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt direction
        [2 / 256, 2 / 256],
        [18 / 256, 18 / 256],
        [16 / 256, 16 / 256],
        [24 / 256, 24 / 256],
        [16 / 256, 16 / 256],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt flags
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt IATs
        [52 / 1000, 52 / 1000],
        [52 / 1000, 52 / 1000],
        [40 / 1000, 40 / 1000],
        [100 / 1000, 500 / 1000],
        [40 / 1000, 40 / 1000],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]  # Pkt size

    # Slow rate attacks with variable IATs
    slowIATsAttacks = [  # noqa: N806
        [0.00000000000001, 0.001],
        [0.0, 0.0],  # Time elapsed, protocol
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt direction
        [2 / 256, 2 / 256],
        [18 / 256, 18 / 256],
        [16 / 256, 16 / 256],
        [24 / 256, 24 / 256],
        [16 / 256, 16 / 256],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt flags
        [0, 1.0],
        [0, 1.0],
        [0, 1.0],
        [0, 1.0],
        [0, 1.0],
        [0, 1.0],
        [0, 1.0],
        [0, 1.0],
        [0, 1.0],
        [0, 1.0],  # Pkt IATs
        [52 / 1000, 52 / 1000],
        [52 / 1000, 52 / 1000],
        [40 / 1000, 40 / 1000],
        [100 / 1000, 500 / 1000],
        [40 / 1000, 40 / 1000],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]  # Pkt size

    # Generic invalid/attack pattern
    invalid = [
        [0.0, 1.0],
        [0.0, 1.0],  # Time elapsed, protocol
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt direction
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt flags
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],  # Pkt IATs
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]  # Pkt size

    return {
        "goodHTTP1": goodHTTP1,
        "goodHTTP2": goodHTTP2,
        "hulk": hulk,
        "slowIATsAttacks": slowIATsAttacks,
        "invalid": invalid,
    }


def create_attack_rectangles(
    feature_names: list[str] = None, attack_pattern: str = "mixed", input_size: int = 42
) -> list[np.ndarray]:
    """
    Create attack rectangles using research-based hyperrectangle definitions.
    This preserves the original paper's contribution of carefully crafted attack patterns.

    Args:
        feature_names: List of feature names (optional, for compatibility)
        attack_pattern: Which attack pattern to use ('hulk', 'goodHTTP1', 'goodHTTP2',
                       'slowIATsAttacks', 'invalid', or 'mixed')
        input_size: Expected input size (default 42 = 2 + 10*4 for pkts_length=10)

    Returns:
        List of attack rectangles as [min_bound, max_bound] arrays
    """
    hyperrects = get_research_hyperrectangles()

    if attack_pattern == "mixed":
        # Use HULK attack pattern as default for adversarial training
        selected_hyperrect = hyperrects["hulk"]
    elif attack_pattern in hyperrects:
        selected_hyperrect = hyperrects[attack_pattern]
    else:
        logger.warning(f"Unknown attack pattern '{attack_pattern}', using 'hulk'")
        selected_hyperrect = hyperrects["hulk"]

    # Convert from research format [min, max] per feature to perturbation bounds
    attack_rects = []
    for i, bounds in enumerate(selected_hyperrect):
        if i >= input_size:
            break
        # Convert absolute bounds to perturbation bounds (relative to input)
        min_bound = bounds[0] if isinstance(bounds, list) and len(bounds) > 0 else -0.1
        max_bound = bounds[1] if isinstance(bounds, list) and len(bounds) > 1 else 0.1

        # For PGD, we need perturbation bounds, so we use the range as delta
        delta_min = min_bound - 0.5  # Assume 0.5 as baseline
        delta_max = max_bound - 0.5

        attack_rects.append(np.array([delta_min, delta_max], dtype=np.float32))

    # Fill remaining features if input_size is larger
    while len(attack_rects) < input_size:
        attack_rects.append(np.array([-0.1, 0.1], dtype=np.float32))

    logger.info(
        f"Created {len(attack_rects)} research-based attack rectangles using '{attack_pattern}' pattern"
    )
    return attack_rects


def generate_hyperrectangle_batch(batch_size: int = 2048, pkts_length: int = 10):
    """
    Generate a batch of hyperrectangles matching the original research methodology.
    This replicates the hyperrectangle generation logic from the original main.py.

    Args:
        batch_size: Size of the batch to generate
        pkts_length: Number of packets per flow (default 10)

    Returns:
        Tuple of (hyperrectangles, labels) as numpy arrays
    """
    import copy

    hyperrects = get_research_hyperrectangles()
    hyperrectangles = []
    hyperrectangles_labels = []

    for _ in range(int((batch_size / 36) + 1)):
        # Add benign traffic patterns (18 samples per batch)
        for _ in range(9):
            hyperrectangles.append(hyperrects["goodHTTP1"])
            hyperrectangles.append(hyperrects["goodHTTP2"])
            hyperrectangles_labels.append(0)  # Benign
            hyperrectangles_labels.append(0)  # Benign

        # Add HULK attack patterns (9 samples per batch)
        for _ in range(9):
            hyperrectangles.append(hyperrects["hulk"])
            hyperrectangles_labels.append(1)  # Attack

        # Add slow IAT attack variants (9 samples per batch)
        for i in range(9):
            temp = copy.deepcopy(hyperrects["slowIATsAttacks"])
            # Modify specific IAT feature (index 23 + i)
            if (23 + i) < len(temp):
                temp[23 + i][0] = 0.06  # Set specific IAT bound
            hyperrectangles.append(temp)
            hyperrectangles_labels.append(1)  # Attack

    # Convert to numpy arrays and trim to exact batch size
    hyperrectangles = np.array(hyperrectangles[:batch_size])
    hyperrectangles_labels = np.array(hyperrectangles_labels[:batch_size])

    logger.info(f"Generated {len(hyperrectangles)} research-based hyperrectangles for training")
    return hyperrectangles, hyperrectangles_labels


def convert_research_hyperrects_to_attack_rects(hyperrectangles: np.ndarray) -> list[np.ndarray]:
    """
    Convert research hyperrectangles to the format expected by PGD attack functions.

    Args:
        hyperrectangles: Research hyperrectangles in original format

    Returns:
        List of attack rectangles for PGD
    """
    if len(hyperrectangles.shape) == 3:
        # Take the first hyperrectangle as template
        template_hyperrect = hyperrectangles[0]
    else:
        template_hyperrect = hyperrectangles

    attack_rects = []
    for feature_bounds in template_hyperrect:
        if isinstance(feature_bounds, list | np.ndarray) and len(feature_bounds) >= 2:
            # Convert to perturbation bounds format
            min_bound = float(feature_bounds[0])
            max_bound = float(feature_bounds[1])
            attack_rects.append(np.array([min_bound, max_bound], dtype=np.float32))
        else:
            # Fallback to default bounds
            attack_rects.append(np.array([-0.1, 0.1], dtype=np.float32))

    return attack_rects
