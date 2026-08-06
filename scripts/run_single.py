#!/usr/bin/env python3
"""
Run a single algorithm on a single instance.

Usage:
    python scripts/run_single.py 10-5-6 GA
    python scripts/run_single.py 20-5-4 IG --time-limit 60
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hfsp.io import InstanceReader
from hfsp.methods.heuristics import neh_heuristic, spt_heuristic, lpt_heuristic
from hfsp.methods.metaheuristics import GeneticAlgorithm, SimulatedAnnealing, IteratedGreedy
from hfsp.visualization.gantt import plot_gantt
from hfsp.utils import RNGManager


def main():
    parser = argparse.ArgumentParser(description="Run a single HFSP solver.")
    parser.add_argument("instance", type=str, help="Instance name (e.g., 10-5-6)")
    parser.add_argument("algorithm", type=str, choices=["NEH", "SPT", "LPT", "GA", "SA", "IG"],
                        help="Algorithm to use")
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Skip Gantt chart")
    parser.add_argument("--output", type=str, default=None, help="Save Gantt chart to file")
    args = parser.parse_args()

    # Load instance
    reader = InstanceReader("Data")
    instance = reader.load(args.instance)
    print(f"Instance: {instance}")
    print()

    # Create RNG
    rng_mgr = RNGManager(base_seed=args.seed)
    rng = rng_mgr.get_generator(0)

    time_limit = args.time_limit if args.time_limit else float("inf")

    # Select algorithm
    name = args.algorithm.upper()
    if name == "NEH":
        sol = neh_heuristic(instance)
    elif name == "SPT":
        sol = spt_heuristic(instance)
    elif name == "LPT":
        sol = lpt_heuristic(instance)
    elif name == "GA":
        ga = GeneticAlgorithm(population_size=80, max_generations=300,
                              rng=rng, time_limit=time_limit)
        sol = ga.solve(instance)
        print(f"Convergence: {len(ga.convergence)} generations")
    elif name == "SA":
        sa = SimulatedAnnealing(max_total_iterations=20000, rng=rng, time_limit=time_limit)
        sol = sa.solve(instance)
        print(f"Convergence: {len(sa.convergence)} iterations")
    elif name == "IG":
        ig = IteratedGreedy(max_iterations=500, rng=rng, time_limit=time_limit)
        sol = ig.solve(instance)
        print(f"Convergence: {len(ig.convergence)} iterations")
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # Print results
    print(f"\n{'='*50}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Makespan:  {sol.makespan:.2f}")
    print(f"Flow Time: {sol.flow_time:.2f}")
    print(f"Method:    {sol.method}")
    print(f"Permutation: {sol.permutation[:10]}..." if len(sol.permutation) > 10
          else f"Permutation: {sol.permutation}")

    # Plot
    if not args.no_plot:
        save_path = args.output
        plot_gantt(sol, save_path=save_path, show=True)


if __name__ == "__main__":
    main()
