"""
Core training function. All optimizers call train_and_evaluate().
To support a different algorithm, update the model instantiation here.
"""

import time
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from training.trial_result import TrialResult


def train_and_evaluate(
    trial: int,
    hyperparams: dict[str, Any],
    env_id: str,
    n_timesteps: int,
    log_dir: str | None = None,
    n_eval_episodes: int = 10,
    seed: int = 42,
) -> TrialResult:
    """
    Train a PPO agent with the given hyperparams and return evaluation metrics.

    Args:
        trial: Trial index (for labelling in results and TensorBoard).
        hyperparams: Dict of PPO hyperparameters (must match PPO_SEARCH_SPACE keys).
        env_id: Gymnasium environment ID (e.g. "CartPole-v1").
        n_timesteps: Total training timesteps.
        log_dir: TensorBoard log directory. None disables logging.
            Callers should use a separate log_dir per optimizer so TensorBoard
            runs are grouped cleanly (e.g. "logs/random/", "logs/evolutionary/").
        n_eval_episodes: Episodes to average over during evaluation.
        seed: Random seed for environment and policy initialisation.
            Defaults to 42 for all trials so that trials differ only in
            hyperparameters, not random variation. Override per-trial if you
            want to average over stochasticity instead.

    Returns:
        TrialResult with reward stats, timesteps used, and wall-clock time.
        wall_time_s covers the full trial: env creation, training, and evaluation.
    """
    tb_log_name = f"trial_{trial}" if log_dir else None

    start = time.time()

    # n_envs=1: CPU-only laptop benchmark — no parallel workers needed
    env = make_vec_env(env_id, n_envs=1, seed=seed)

    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=hyperparams["learning_rate"],
            n_steps=hyperparams["n_steps"],
            batch_size=hyperparams["batch_size"],
            gamma=hyperparams["gamma"],
            gae_lambda=hyperparams["gae_lambda"],
            ent_coef=hyperparams["ent_coef"],
            n_epochs=hyperparams["n_epochs"],
            tensorboard_log=log_dir,
            verbose=0,
            seed=seed,
        )

        model.learn(total_timesteps=n_timesteps, tb_log_name=tb_log_name)

        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
    finally:
        env.close()

    wall_time_s = time.time() - start

    return TrialResult(
        trial=trial,
        hyperparams=hyperparams,
        mean_reward=float(mean_reward),
        std_reward=float(std_reward),
        total_timesteps_used=n_timesteps,
        wall_time_s=wall_time_s,
    )
