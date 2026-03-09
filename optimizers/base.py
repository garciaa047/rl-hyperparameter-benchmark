"""
Abstract base class for all hyperparameter optimizers.
Every optimizer must implement optimize() and return list[TrialResult].

To add a new optimizer:
  1. Subclass BaseOptimizer
  2. Implement optimize()
  3. Register it in run_benchmark.py's OPTIMIZER_MAP
"""

from abc import ABC, abstractmethod

from training.trial_result import TrialResult


class BaseOptimizer(ABC):
    """
    Shared interface for all hyperparameter optimizers.

    All optimizers receive the same inputs and return the same output type,
    making them directly interchangeable in the benchmark runner.
    """

    @abstractmethod
    def optimize(
        self,
        env_id: str,
        n_trials: int,
        n_timesteps: int,
        log_dir: str | None = None,
    ) -> list[TrialResult]:
        """
        Run hyperparameter search and return all trial results.

        Args:
            env_id: Gymnasium environment ID (e.g. "CartPole-v1").
            n_trials: Number of configurations to evaluate.
                      Note: some optimizers (e.g. Evolutionary) determine trial
                      count from their internal structure (population_size *
                      n_generations) rather than this argument. Callers should
                      not assume len(results) == n_trials; the actual count
                      depends on the optimizer.
            n_timesteps: Training timesteps per trial.
            log_dir: TensorBoard log directory. None disables logging.

        Returns:
            List of TrialResult, one per trial evaluated.
        """
        ...
