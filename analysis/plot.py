"""
Plotting utilities for benchmark analysis.
Called by analyze.py after a benchmark run.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir: str = "results") -> pd.DataFrame:
    """
    Load and merge all CSVs from results_dir into a single DataFrame.
    Infers optimizer name and env from filename.
    """
    dfs = []
    for path in Path(results_dir).glob("*.csv"):
        parts = path.stem.split("_")
        optimizer = parts[0]
        df = pd.read_csv(path)
        df["optimizer"] = optimizer
        df["source_file"] = path.name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in '{results_dir}/'")

    return pd.concat(dfs, ignore_index=True)


def plot_mean_reward(df: pd.DataFrame, output_dir: str = "analysis/output"):
    """Bar chart: mean reward per optimizer (with std error bars)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary = df.groupby("optimizer")["mean_reward"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary, x="optimizer", y="mean", yerr=summary["std"], ax=ax, palette="Set2")
    ax.set_title("Mean Reward by Optimizer")
    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Optimizer")
    plt.tight_layout()
    path = Path(output_dir) / "mean_reward.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_wall_time(df: pd.DataFrame, output_dir: str = "analysis/output"):
    """Box plot: wall time distribution per optimizer."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="optimizer", y="wall_time_s", palette="Set2", ax=ax)
    ax.set_title("Wall-Clock Time per Trial by Optimizer")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Optimizer")
    plt.tight_layout()
    path = Path(output_dir) / "wall_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_reward_over_trials(df: pd.DataFrame, output_dir: str = "analysis/output"):
    """Line chart: reward progression across trials per optimizer."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for optimizer, group in df.groupby("optimizer"):
        group_sorted = group.sort_values("trial")
        ax.plot(group_sorted["trial"], group_sorted["mean_reward"], marker="o", label=optimizer)
    ax.set_title("Reward Progression Over Trials")
    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Trial")
    ax.legend()
    plt.tight_layout()
    path = Path(output_dir) / "reward_over_trials.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")