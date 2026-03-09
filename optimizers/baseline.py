"""
Baseline optimizer: runs PPO once with SB3 default hyperparameters.
No search is performed — this is the control condition for comparison.
"""

from training.trainer import train_and_evaluate
from training.search_space import sample_ppo_defaults
from training.trial_result import TrialResult
from optimizers.base import BaseOptimizer


class BaselineOptimizer(BaseOptimizer):
    """
    Runs a single trial with SB3 PPO default hyperparameters.

    Always returns exactly one TrialResult regardless of n_trials.
    Use this as the control condition to compare against search strategies.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def optimize(
        self,
        env_id: str,
        n_trials: int,
        n_timesteps: int,
        log_dir: str | None = None,
    ) -> list[TrialResult]:
        hyperparams = sample_ppo_defaults()
        result = train_and_evaluate(
            trial=1,
            hyperparams=hyperparams,
            env_id=env_id,
            n_timesteps=n_timesteps,
            log_dir=log_dir,
            seed=self.seed,
        )
        return [result]
