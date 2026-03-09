"""
Utility to sample a single hyperparameter configuration from a search space dict.
Used by RandomSearchOptimizer and EvolutionaryOptimizer.
"""

import numpy as np
from typing import Any

from training.search_space import PPO_SEARCH_SPACE


def sample_random_config(
    space: dict[str, dict[str, Any]] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Sample one config by drawing each parameter independently at random.

    Args:
        space: Search space definition. Defaults to PPO_SEARCH_SPACE.
        rng: numpy random Generator for reproducibility. Creates a new one
             if not provided (non-reproducible).

    Returns:
        Dict of hyperparameter name -> sampled value.
    """
    if space is None:
        space = PPO_SEARCH_SPACE
    if rng is None:
        rng = np.random.default_rng()

    config: dict[str, Any] = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "log_float":
            log_val = rng.uniform(np.log(spec["low"]), np.log(spec["high"]))
            config[name] = float(np.exp(log_val))
        elif t == "float":
            config[name] = float(rng.uniform(spec["low"], spec["high"]))
        elif t == "int":
            config[name] = int(rng.integers(spec["low"], spec["high"] + 1))
        elif t == "categorical":
            # .item() converts numpy scalar to Python native type.
            # Assumes choices contains numeric or string values compatible with numpy arrays.
            config[name] = rng.choice(spec["choices"]).item()
        else:
            raise ValueError(f"Unknown parameter type '{t}' for '{name}'")
    return config
