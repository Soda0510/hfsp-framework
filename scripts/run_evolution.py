"""
P2 verification: evolutionary loop with RANDOM mutation (baseline).

Establishes that the population / evaluate / select / mutate loop works and
converges, before the LLM mutation is plugged in (P3).  Also computes the
"random baseline" fitness that the LLM will later be compared against.

Usage:
    .venv/bin/python scripts/run_evolution.py [--instances 10-3-3,10-3-4,10-3-6]
                                              [--pop 8] [--gen 6] [--seed 123]
"""

import argparse
import sys

import numpy as np

from hfsp.io import InstanceReader
from hfsp.llm import (
    make_fitness, per_instance_best, EvolutionarySearch,
    random_spec, perturb_spec,
)
from hfsp.llm.seeds import default_seed_specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", default="10-3-3,10-3-4,10-3-6")
    parser.add_argument("--pop", type=int, default=6)
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    reader = InstanceReader("Data")
    instances = [reader.load(n) for n in args.instances.split(",")]
    print(f"Training instances: {[i.name for i in instances]}")

    seeds = default_seed_specs()

    # Equal time budget for every evaluation (standard fair-comparison choice
    # in metaheuristic research; also keeps runtime bounded).
    TIME_BUDGET = 1.0  # seconds per (spec, instance, run)

    # Reference per instance = best makespan among the seed algorithms.
    refs = per_instance_best(instances, seeds, n_runs=1, seed_base=0,
                             time_limit=TIME_BUDGET)
    print(f"Per-instance references (best of seeds, {TIME_BUDGET}s budget): {refs}")

    fitness = make_fitness(instances, n_runs=1, references=refs, seed_base=0,
                           time_limit=TIME_BUDGET)

    # Random baseline: mean fitness of fresh random specs (upper bound to beat).
    rng = np.random.default_rng(args.seed)
    random_fits = [fitness(random_spec(rng, "baseline")) for _ in range(6)]
    random_mean = float(np.mean(random_fits))
    print(f"\nRandom baseline: mean RPD = {random_mean:+.2f}% "
          f"(best {min(random_fits):+.2f}%, worst {max(random_fits):+.2f}%)")

    seed_fits = {s.name: fitness(s) for s in seeds}
    print(f"Seed fitness (RPD%): {seed_fits}")
    best_seed_fit = min(seed_fits.values())
    print(f"Best seed: {best_seed_fit:+.2f}%")

    # ---- Evolutionary search with random mutation ----
    print(f"\nEvolutionary search (pop={args.pop}, gen={args.gen}, random mutation):")
    evo = EvolutionarySearch(
        fitness_fn=fitness,
        mutator=perturb_spec,           # random mutation for P2
        population_size=args.pop,
        elite_size=2,
        generations=args.gen,
        seed=args.seed,
        seed_specs=seeds,
    )
    result = evo.run()

    print(f"\n{'='*60}")
    print(f"Best evolved spec : {result['best_spec'].name}")
    print(f"Best evolved RPD  : {result['best_fitness']:+.2f}%")
    print(f"Convergence       : {[f'{h:+.2f}' for h in result['history']]}")
    print(f"Spec              : {result['best_spec'].to_dict()}")

    # ---- Checks ----
    ok = True
    if result["best_fitness"] > best_seed_fit + 1e-6:
        print(f"  FAIL: evolved ({result['best_fitness']:+.2f}%) "
              f"worse than best seed ({best_seed_fit:+.2f}%)")
        ok = False
    else:
        print(f"  OK  : evolved <= best seed  ({result['best_fitness']:+.2f}% <= {best_seed_fit:+.2f}%)")

    if result["best_fitness"] > random_mean + 1e-6:
        print(f"  FAIL: evolved ({result['best_fitness']:+.2f}%) "
              f"not better than random baseline ({random_mean:+.2f}%)")
        ok = False
    else:
        print(f"  OK  : evolved < random baseline "
              f"({result['best_fitness']:+.2f}% < {random_mean:+.2f}%)")

    print(f"\n{'P2 PASSED' if ok else 'P2 FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
