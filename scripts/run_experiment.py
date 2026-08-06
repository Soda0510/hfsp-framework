#!/usr/bin/env python3
"""
Run a batch experiment: multiple algorithms on multiple instances.

Usage:
    python scripts/run_experiment.py --instances "10-*" --algorithms NEH,GA,IG --runs 5
    python scripts/run_experiment.py --all --runs 10 --time-limit 60
"""

import sys
import os
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hfsp.io import InstanceReader
from hfsp.experiment import (
    ExperimentConfig, AlgorithmConfig, ExperimentRunner
)


def main():
    parser = argparse.ArgumentParser(description="Run batch HFSP experiments.")
    parser.add_argument("--instances", type=str, default="",
                        help="Comma-separated instance patterns (e.g., '10-*,20-3-*')")
    parser.add_argument("--algorithms", type=str, default="NEH,GA,SA,IG",
                        help="Comma-separated algorithm names")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of independent runs per config")
    parser.add_argument("--seed", type=int, default=20240101, help="Base random seed")
    parser.add_argument("--time-limit", type=float, default=60.0,
                        help="Time limit per run (seconds)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--all", action="store_true",
                        help="Run all instances (overrides --instances)")
    parser.add_argument("--ga-generations", type=int, default=300,
                        help="GA generations")
    parser.add_argument("--sa-iterations", type=int, default=20000,
                        help="SA max iterations")
    parser.add_argument("--ig-iterations", type=int, default=500,
                        help="IG max iterations")
    args = parser.parse_args()

    # Parse arguments
    algo_names = [a.strip() for a in args.algorithms.split(",")]
    instance_patterns = []
    if not args.all and args.instances:
        instance_patterns = [p.strip() for p in args.instances.split(",")]
    elif args.all:
        # All instances
        reader = InstanceReader("Data")
        instance_patterns = reader.list_instances()
    else:
        # Default: small instances
        instance_patterns = ["10-*", "20-*"]

    # Build algorithm configs
    algo_configs = []
    for name in algo_names:
        algo_configs.append(AlgorithmConfig(
            name=name,
            population_size=80,
            max_generations=args.ga_generations,
            max_total_iterations=args.sa_iterations,
            ig_iterations=args.ig_iterations,
            time_limit=args.time_limit,
        ))

    # Build experiment config
    config = ExperimentConfig(
        name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description=f"Algorithms: {algo_names}, Instances: {instance_patterns}",
        instances=instance_patterns,
        algorithms=algo_configs,
        num_runs=args.runs,
        seed_base=args.seed,
        output_dir=args.output,
    )

    # Run
    reader = InstanceReader("Data")
    runner = ExperimentRunner(config, reader)
    print(f"Running experiment: {len(algo_configs)} algorithms × "
          f"{len(config._resolve_instances() if hasattr(runner, '_resolve_instances') else '?' )} instances × "
          f"{args.runs} runs")
    results = runner.run()

    # Save
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "results.csv")
    results.save_csv(csv_path)

    # Summary
    summary = results.compute_summary()
    summary_path = os.path.join(args.output, "summary.csv")
    summary.to_csv(summary_path, index=False)

    # RPD
    rpd_df = results.compute_rpd()
    rpd_path = os.path.join(args.output, "rpd.csv")
    rpd_df.to_csv(rpd_path, index=False)

    print(f"\nResults saved to {args.output}/")
    print(f"  - results.csv ({len(results.results)} runs)")
    print(f"  - summary.csv")
    print(f"  - rpd.csv")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY (mean ± std makespan)")
    print(f"{'='*80}")
    summary_pivot = summary.pivot(
        index="instance", columns="algorithm", values="mean_makespan"
    )
    print(summary_pivot.to_string(float_format="%.1f"))


if __name__ == "__main__":
    main()
