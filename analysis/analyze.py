"""
Post-benchmark analysis script.
Run this after run_benchmark.py to generate comparison plots.

Usage:
    python analyze.py
    python analyze.py --results-dir results --output-dir analysis/output
"""

import argparse
from analysis.plot import load_results, plot_mean_reward, plot_wall_time, plot_reward_over_trials


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse benchmark results and generate plots")
    parser.add_argument("--results-dir", default="results", help="Directory containing CSV results")
    parser.add_argument("--output-dir", default="analysis/output", help="Directory to save plots")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\nLoading results from '{args.results_dir}/'...")

    df = load_results(args.results_dir)
    print(f"  Loaded {len(df)} trial rows from {df['source_file'].nunique()} file(s)")
    print(f"  Optimizers found: {sorted(df['optimizer'].unique())}\n")

    print("Generating plots...")
    plot_mean_reward(df, args.output_dir)
    plot_wall_time(df, args.output_dir)
    plot_reward_over_trials(df, args.output_dir)

    print(f"\nDone. Plots saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()