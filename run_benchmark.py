"""
RL Hyperparameter Benchmark runner.

Usage:
    python run_benchmark.py
    python run_benchmark.py --env LunarLander-v2 --optimizers random evolutionary --n-trials 5
    python run_benchmark.py --help
"""

import argparse
import csv
import datetime
from pathlib import Path

from optimizers.baseline import BaselineOptimizer
from optimizers.random_search import RandomSearchOptimizer
from optimizers.evolutionary import EvolutionaryOptimizer
from optimizers.optuna_search import OptunaOptimizer

OPTIMIZER_MAP = {
    "baseline":    BaselineOptimizer,
    "random":      RandomSearchOptimizer,
    "evolutionary": EvolutionaryOptimizer,
    "optuna":      OptunaOptimizer,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PPO hyperparameter optimizers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env", default="CartPole-v1",
        help="Gymnasium environment ID",
    )
    parser.add_argument(
        "--optimizers", nargs="+",
        default=list(OPTIMIZER_MAP.keys()),
        choices=list(OPTIMIZER_MAP.keys()),
        help="Optimizers to run",
    )
    parser.add_argument(
        "--n-trials", type=int, default=10,
        help="Trials per optimizer (ignored by evolutionary, which uses pop*gen)",
    )
    parser.add_argument(
        "--n-timesteps", type=int, default=50000,
        help="PPO training timesteps per trial",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory to save CSV result files",
    )
    parser.add_argument(
        "--log-dir", default="logs",
        help="Root TensorBoard log directory (subdirs created per optimizer)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed passed to all optimizers",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true",
        help="Disable TensorBoard logging",
    )
    return parser.parse_args()


def save_results(results, optimizer_name: str, env_id: str, output_dir: str) -> Path:
    """Save trial results to a timestamped CSV file. Returns the saved path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_slug = env_id.lower().replace("-", "").replace("_", "").replace("/", "")
    filename = f"{optimizer_name}_{env_slug}_{timestamp}.csv"
    filepath = Path(output_dir) / filename

    rows = [r.to_dict() for r in results]
    if not rows:
        print(f"  Warning: no results to save for {optimizer_name}")
        return filepath

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved  -> {filepath}")
    return filepath


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  RL Hyperparameter Benchmark")
    print(f"  Environment  : {args.env}")
    print(f"  Optimizers   : {args.optimizers}")
    print(f"  n_trials     : {args.n_trials}  (evolutionary uses pop*gen)")
    print(f"  n_timesteps  : {args.n_timesteps}")
    print(f"  Seed         : {args.seed}")
    print(f"  TensorBoard  : {'disabled' if args.no_tensorboard else args.log_dir}")
    print(f"{'='*60}\n")

    summary: list[dict] = []

    for name in args.optimizers:
        print(f"\n{'-'*50}")
        print(f"  Running optimizer: {name}")
        print(f"{'-'*50}")

        optimizer_cls = OPTIMIZER_MAP[name]
        optimizer = optimizer_cls(seed=args.seed)

        # Per-optimizer TensorBoard subdir keeps runs grouped in the dashboard
        if args.no_tensorboard:
            log_dir = None
        else:
            log_dir = str(Path(args.log_dir) / name)

        results = optimizer.optimize(
            env_id=args.env,
            n_trials=args.n_trials,
            n_timesteps=args.n_timesteps,
            log_dir=log_dir,
        )

        if results:
            best = max(results, key=lambda r: r.mean_reward)
            avg_time = sum(r.wall_time_s for r in results) / len(results)
            print(f"\n  Trials completed : {len(results)}")
            print(f"  Best reward      : {best.mean_reward:.2f} +/- {best.std_reward:.2f}")
            print(f"  Avg trial time   : {avg_time:.1f}s")
            save_results(results, name, args.env, args.output_dir)
            summary.append({
                "optimizer": name,
                "n_trials": len(results),
                "best_reward": best.mean_reward,
                "best_std": best.std_reward,
                "avg_trial_time_s": avg_time,
            })
        else:
            print(f"  Warning: {name} returned no results")

    # Final summary table
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY  ({args.env})")
    print(f"{'='*60}")
    print(f"  {'Optimizer':<16} {'Trials':>7} {'Best Reward':>12} {'Avg Time(s)':>12}")
    print(f"  {'-'*16} {'-'*7} {'-'*12} {'-'*12}")
    for row in summary:
        print(f"  {row['optimizer']:<16} {row['n_trials']:>7} {row['best_reward']:>12.2f} {row['avg_trial_time_s']:>12.1f}")
    print(f"{'='*60}")

    print(f"\nRun 'python analyze.py' to generate comparison plots.")
    if not args.no_tensorboard:
        print(f"TensorBoard: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
