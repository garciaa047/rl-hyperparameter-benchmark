from dataclasses import dataclass
from typing import Any


@dataclass
class TrialResult:
    """
    Result of a single hyperparameter trial.

    Attributes:
        trial: 1-based trial index within the optimizer run.
        hyperparams: Dict of hyperparameter names to sampled values.
        mean_reward: Mean episodic reward over eval episodes.
        std_reward: Standard deviation of episodic reward over eval episodes.
        total_timesteps_used: Environment steps consumed during training.
        wall_time_s: Real elapsed wall-clock time in seconds.
    """

    trial: int
    hyperparams: dict[str, Any]
    mean_reward: float
    std_reward: float
    total_timesteps_used: int
    wall_time_s: float

    def __post_init__(self):
        # Defensive copy: prevent callers from mutating hyperparams after construction
        self.hyperparams = dict(self.hyperparams)

    def to_dict(self) -> dict[str, Any]:
        """
        Flatten into a dict suitable for a single CSV row.

        Scalar fields come first, then all hyperparameter keys are appended.
        Raises ValueError if any hyperparam key clashes with a scalar field name,
        which would otherwise silently corrupt CSV output.
        """
        scalar_keys = {"trial", "mean_reward", "std_reward", "total_timesteps_used", "wall_time_s"}
        clashes = scalar_keys & self.hyperparams.keys()
        if clashes:
            raise ValueError(f"Hyperparam keys clash with TrialResult scalar fields: {clashes}")
        row: dict[str, Any] = {
            "trial": self.trial,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "total_timesteps_used": self.total_timesteps_used,
            "wall_time_s": self.wall_time_s,
        }
        row.update(self.hyperparams)
        return row
