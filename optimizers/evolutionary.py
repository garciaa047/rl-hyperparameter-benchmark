"""
Evolutionary optimizer: simple genetic algorithm for hyperparameter search.

Algorithm:
  1. Initialise population of `population_size` random configs
  2. Evaluate each config -> get fitness (mean_reward)
  3. Select top-K survivors (elitism)
  4. Mutate survivors to fill next generation
  5. Repeat for `n_generations` generations

Total trials = population_size * n_generations.
The n_trials argument from BaseOptimizer is ignored.
"""

import numpy as np
from typing import Any

from training.trainer import train_and_evaluate
from training.sampler import sample_random_config
from training.search_space import PPO_SEARCH_SPACE
from training.trial_result import TrialResult
from optimizers.base import BaseOptimizer


def mutate_config(
    config: dict[str, Any],
    space: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    mutation_rate: float = 0.3,
    mutation_strength: float = 0.2,
) -> dict[str, Any]:
    """
    Apply random mutation to a hyperparameter config.

    Each parameter mutates independently with probability `mutation_rate`.

    Mutation strategy per type:
      - log_float: Gaussian noise in log-space, clipped to [log(low), log(high)]
      - float: Gaussian noise scaled by mutation_strength * range, clipped to [low, high]
      - int / categorical: re-sample uniformly from valid values

    Args:
        config: Current hyperparameter config to mutate.
        space: Search space defining valid ranges/choices per parameter.
        rng: numpy random Generator (caller owns this for reproducibility).
        mutation_rate: Probability that each parameter is mutated. Default 0.3.
        mutation_strength: Noise scale as a fraction of the parameter range. Default 0.2.

    Precondition:
        For parameters with type "log_float", config[name] must be > 0.
        Passing 0.0 (e.g. from sample_ppo_defaults() ent_coef) will produce
        np.log(0) = -inf and corrupt the mutated value.

    Returns:
        New config dict (copy of input with some values mutated).
    """
    mutated = config.copy()
    for name, spec in space.items():
        if rng.random() > mutation_rate:
            continue

        t = spec["type"]
        if t == "log_float":
            log_low = np.log(spec["low"])
            log_high = np.log(spec["high"])
            noise = rng.normal(0.0, mutation_strength * (log_high - log_low))
            new_log_val = np.clip(np.log(config[name]) + noise, log_low, log_high)
            mutated[name] = float(np.clip(np.exp(new_log_val), spec["low"], spec["high"]))
        elif t == "float":
            noise = rng.normal(0.0, mutation_strength * (spec["high"] - spec["low"]))
            mutated[name] = float(np.clip(config[name] + noise, spec["low"], spec["high"]))
        elif t == "int":
            mutated[name] = int(rng.integers(spec["low"], spec["high"] + 1))
        elif t == "categorical":
            mutated[name] = rng.choice(spec["choices"]).item()

    return mutated


class EvolutionaryOptimizer(BaseOptimizer):
    """
    Simple genetic algorithm optimizer.

    Runs population_size * n_generations total trials.
    The n_trials argument from BaseOptimizer.optimize() is ignored.
    """

    def __init__(
        self,
        population_size: int = 6,
        n_generations: int = 3,
        top_k: int = 3,
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.2,
        seed: int = 42,
    ):
        """
        Args:
            population_size: Number of configs evaluated per generation.
            n_generations: Number of generations to run.
            top_k: Number of survivors per generation (elitism).
                Setting top_k=1 degrades the algorithm to a local mutation
                search — diversity relies entirely on mutation_rate/strength.
            mutation_rate: Probability each parameter is mutated per generation.
            mutation_strength: Gaussian noise scale as fraction of param range.
            seed: Random seed for reproducibility.
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.top_k = min(top_k, population_size)
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.seed = seed

    def optimize(
        self,
        env_id: str,
        n_trials: int,
        n_timesteps: int,
        log_dir: str | None = None,
    ) -> list[TrialResult]:
        total_trials = self.population_size * self.n_generations
        print(f"  [Evolutionary] n_trials={n_trials} ignored; running {total_trials} trials ({self.population_size} pop × {self.n_generations} gen)")
        rng = np.random.default_rng(self.seed)
        all_results: list[TrialResult] = []
        trial_counter = 1

        # Initialise first generation with random configs
        population = [sample_random_config(rng=rng) for _ in range(self.population_size)]

        for generation in range(self.n_generations):
            print(f"  [Evolutionary] Generation {generation + 1}/{self.n_generations}")
            gen_results: list[TrialResult] = []

            for idx, config in enumerate(population):
                print(f"  [Evolutionary] Trial {trial_counter}/{total_trials} | lr={config['learning_rate']:.2e} n_steps={config['n_steps']}")
                result = train_and_evaluate(
                    trial=trial_counter,
                    hyperparams=config,
                    env_id=env_id,
                    n_timesteps=n_timesteps,
                    log_dir=log_dir,
                    seed=self.seed,
                )
                print(f"  [Evolutionary] Trial {trial_counter}/{total_trials} | reward={result.mean_reward:.2f} ± {result.std_reward:.2f} | time={result.wall_time_s:.1f}s")
                gen_results.append(result)
                all_results.append(result)
                trial_counter += 1

            # Select top-K survivors by mean_reward (elitism)
            gen_results.sort(key=lambda r: r.mean_reward, reverse=True)
            survivors = [r.hyperparams for r in gen_results[: self.top_k]]
            best = gen_results[0]
            print(f"  [Evolutionary] Generation {generation + 1} best: reward={best.mean_reward:.2f}")

            # Build next generation by mutating survivors (skip after last generation)
            if generation < self.n_generations - 1:
                population = []
                for i in range(self.population_size):
                    parent = survivors[i % len(survivors)]
                    child = mutate_config(parent, PPO_SEARCH_SPACE, rng, self.mutation_rate, self.mutation_strength)
                    population.append(child)

        return all_results
