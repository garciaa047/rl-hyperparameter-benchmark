"""
Random search optimizer: evaluates hyperparameter configs sampled uniformly
at random from the search space.
"""

import numpy as np

from training.trainer import train_and_evaluate
from training.sampler import sample_random_config
from training.trial_result import TrialResult
from optimizers.base import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    """
    Samples hyperparameter configurations uniformly at random from PPO_SEARCH_SPACE
    and evaluates each independently.

    Each trial is completely independent — there is no learning between trials.
    Use this as a baseline search strategy to compare against smarter optimizers.
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
        # seed is fixed per trial so that differences in reward reflect
        # hyperparameter quality, not environment/init randomness
        rng = np.random.default_rng(self.seed)
        results: list[TrialResult] = []

        for i in range(n_trials):
            hyperparams = sample_random_config(rng=rng)
            print(f"  [RandomSearch] Trial {i + 1}/{n_trials} | lr={hyperparams['learning_rate']:.2e} n_steps={hyperparams['n_steps']}")
            result = train_and_evaluate(
                trial=i + 1,
                hyperparams=hyperparams,
                env_id=env_id,
                n_timesteps=n_timesteps,
                log_dir=log_dir,
                seed=self.seed,
            )
            print(f"  [RandomSearch] Trial {i + 1}/{n_trials} | reward={result.mean_reward:.2f} ± {result.std_reward:.2f} | time={result.wall_time_s:.1f}s")
            results.append(result)

        return results
