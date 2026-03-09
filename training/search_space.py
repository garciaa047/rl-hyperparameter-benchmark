"""
PPO hyperparameter search space definition.
To support a different algorithm, replace this file and update trainer.py.

Each entry describes how to sample the parameter:
  - type: "log_float" | "float" | "int" | "categorical"
  - low / high: bounds for numeric types
  - choices: list for categorical types

Constraint: batch_size must be <= n_steps (SB3 requirement).
batch_size choices are capped at 128 (the smallest n_steps choice) so any
sampled combination is valid without a runtime check.
"""

from typing import Any

PPO_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "log_float", "low": 1e-5,  "high": 1e-2},
    "n_steps":       {"type": "categorical", "choices": [128, 256, 512, 1024, 2048]},
    "batch_size":    {"type": "categorical", "choices": [32, 64, 128]},
    "gamma":         {"type": "float",       "low": 0.9,  "high": 0.9999},
    "gae_lambda":    {"type": "float",       "low": 0.8,  "high": 0.99},
    "ent_coef":      {"type": "log_float",   "low": 1e-8, "high": 0.1},
    "n_epochs":      {"type": "int",         "low": 3,    "high": 30},
}


def sample_ppo_defaults() -> dict[str, Any]:
    """
    Return SB3 PPO default hyperparameters.

    Note: ent_coef is 0.0 (SB3's actual default), which is below the search
    space lower bound of 1e-8. This is intentional — the baseline optimizer
    uses the true SB3 default. Do NOT pass this config to mutate_config(),
    as np.log(0.0) = -inf will produce undefined behaviour.
    """
    return {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "n_epochs": 10,
    }
