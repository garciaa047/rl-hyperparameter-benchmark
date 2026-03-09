"""
Optuna optimizer: uses Tree-structured Parzen Estimator (TPE) to guide search.

Optuna builds a probabilistic model of which hyperparameters led to good rewards
and uses it to suggest promising configs for the next trial. This is a form of
Bayesian optimisation.
"""

import optuna
from typing import Any

from training.trainer import train_and_evaluate
from training.search_space import PPO_SEARCH_SPACE
from training.trial_result import TrialResult
from optimizers.base import BaseOptimizer

# Suppress Optuna's verbose per-trial output — we print our own
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_from_space(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Translate PPO_SEARCH_SPACE spec into Optuna suggest_* calls.

    Maps:
      log_float  -> trial.suggest_float(..., log=True)
      float      -> trial.suggest_float(...)
      int        -> trial.suggest_int(...)
      categorical -> trial.suggest_categorical(...)
    """
    config: dict[str, Any] = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "log_float":
            config[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif t == "float":
            config[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif t == "int":
            config[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "categorical":
            config[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown parameter type '{t}' for '{name}'")
    return config


class OptunaOptimizer(BaseOptimizer):
    """
    Bayesian hyperparameter optimizer using Optuna's TPE sampler.

    Builds a probabilistic model over trials to suggest increasingly
    promising hyperparameter configs. Generally more sample-efficient
    than random search after the first few warm-up trials.
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
        results: list[TrialResult] = []

        def objective(trial: optuna.Trial) -> float:
            hyperparams = _suggest_from_space(trial, PPO_SEARCH_SPACE)
            # trial.number is 0-based and increments even for failed/pruned trials.
            # In sequential mode with no failures this equals completion order.
            trial_num = trial.number + 1  # 1-based for display and TrialResult
            print(f"  [Optuna] Trial {trial_num}/{n_trials} | lr={hyperparams['learning_rate']:.2e} n_steps={hyperparams['n_steps']}")
            result = train_and_evaluate(
                trial=trial_num,
                hyperparams=hyperparams,
                env_id=env_id,
                n_timesteps=n_timesteps,
                log_dir=log_dir,
                seed=self.seed,
            )
            print(f"  [Optuna] Trial {trial_num}/{n_trials} | reward={result.mean_reward:.2f} ± {result.std_reward:.2f} | time={result.wall_time_s:.1f}s")
            results.append(result)
            return result.mean_reward

        # In-memory study: results are not persisted between runs.
        # To enable resume/dashboard support, pass a storage= argument.
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        # n_jobs=1 required: results are collected via closure append;
        # parallel execution would produce non-deterministic ordering.
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        results.sort(key=lambda r: r.trial)
        return results
