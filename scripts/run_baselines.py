"""
P5b: run baseline metaheuristics on the 480 testbed under a UNIFORM time budget.

Each algorithm gets t = rho * n * m / 1000 seconds per (instance, run), the
standard fair-comparison protocol (Ruiz-Stützle / Pan et al. convention).
Constructive heuristics (NEH/SPT/LPT/NEH+LS) are deterministic and fast.

Usage:
    .venv/bin/python scripts/run_baselines.py --sample small --rho 10 --runs 1
    .venv/bin/python scripts/run_baselines.py --instances instancia_10_5_1,instancia_20_10_1
"""

import argparse
import sys
import time
from typing import Dict, List

import numpy as np

from hfsp.io import SevilleReader, SevilleReference
from hfsp.core.decoder import ListSchedulingDecoder
from hfsp.methods.heuristics import neh_heuristic, spt_heuristic, lpt_heuristic
from hfsp.methods.metaheuristics import (
    GeneticAlgorithm, SimulatedAnnealing, IteratedGreedy, DiscretePSO,
)
from hfsp.methods.operators import local_search

ROOT = "benchmarks/seville"
REF = "benchmarks/seville/references/UpperBounds_01_April_2019.xlsx"

# (name, builder) — builder(rng, time_limit) -> Method-like object
# Constructive ones ignore the time budget.


def _build_neh(rng, tl):
    from types import SimpleNamespace
    return SimpleNamespace(solve=lambda inst: neh_heuristic(inst, ListSchedulingDecoder(), rng))


def _build_spt(rng, tl):
    from types import SimpleNamespace
    return SimpleNamespace(solve=lambda inst: spt_heuristic(inst, ListSchedulingDecoder(), rng))


def _build_lpt(rng, tl):
    from types import SimpleNamespace
    return SimpleNamespace(solve=lambda inst: lpt_heuristic(inst, ListSchedulingDecoder(), rng))


def _build_neh_ls(rng, tl):
    from types import SimpleNamespace
    def solve(inst):
        sol = neh_heuristic(inst, ListSchedulingDecoder(), rng)
        return local_search(sol, ListSchedulingDecoder(), max_iterations=200,
                            strategy="first_improvement", rng=rng)
    return SimpleNamespace(solve=solve)


def _build_ga(rng, tl):
    return GeneticAlgorithm(population_size=60, crossover_prob=0.8, mutation_prob=0.3,
                            max_generations=2000, elite_size=2, tournament_size=2,
                            rng=rng, time_limit=tl)


def _build_sa(rng, tl):
    return SimulatedAnnealing(initial_temperature=100.0, cooling_rate=0.97,
                              max_iterations=200, max_total_iterations=500000,
                              rng=rng, time_limit=tl)


def _build_ig(rng, tl):
    return IteratedGreedy(destruction_size=None, temperature=None,
                          max_iterations=50000, use_local_search=True,
                          rng=rng, time_limit=tl)


def _build_dpso(rng, tl):
    return DiscretePSO(swarm_size=40, max_iterations=2000, w=0.5, c1=0.3, c2=0.2,
                       rng=rng, time_limit=tl)


BASELINES: Dict[str, callable] = {
    "NEH": _build_neh,
    "SPT": _build_spt,
    "LPT": _build_lpt,
    "NEH+LS": _build_neh_ls,
    "GA": _build_ga,
    "SA": _build_sa,
    "IG": _build_ig,
    "DPSO": _build_dpso,
}


def _budget(inst, rho: float) -> float:
    return rho * inst.num_jobs * inst.total_machines / 1000.0   # seconds


def run_sample(instance_names: List[str], rho: float, runs: int, seed: int):
    reader = SevilleReader(ROOT)
    ref = SevilleReference(REF)
    rng_master = np.random.default_rng(seed)

    # RPD per algorithm: list of per-instance mean RPD
    results: Dict[str, List[float]] = {name: [] for name in BASELINES}

    for name in instance_names:
        inst = reader.load(name)
        bk = ref.best_known(name)
        t = _budget(inst, rho)
        for algo, builder in BASELINES.items():
            best_over_runs = []
            for r in range(runs):
                rng = np.random.default_rng(seed + hash(name) % 10000 + r)
                method = builder(rng, t)
                sol = method.solve(inst)
                best_over_runs.append(sol.makespan)
            mean_ms = float(np.mean(best_over_runs))
            results[algo].append((mean_ms - bk) / bk * 100.0)

    print(f"instances: {len(instance_names)} | rho={rho} (budget=rho*n*m/1000s) | runs={runs}")
    print(f"\n{'algorithm':<10}{'mean RPD':>10}{'min RPD':>10}{'max RPD':>10}"
          f"{'  #hitsBK':>9}")
    print("-" * 50)
    for algo, vals in results.items():
        hits = sum(1 for v in vals if v < 1e-6)
        print(f"{algo:<10}{np.mean(vals):>10.2f}{np.min(vals):>10.2f}"
              f"{np.max(vals):>10.2f}{hits:>9}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=str, default=None,
                        help="comma-separated instance names")
    parser.add_argument("--sample", type=str, default=None,
                        help="'small' | 'big' | 'all' — representative stratified sample")
    parser.add_argument("--rho", type=float, default=10.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()

    reader = SevilleReader(ROOT)
    if args.instances:
        names = [n.strip() for n in args.instances.split(",")]
    elif args.sample:
        # stratified sample: pick 2 per (jobs, stages) combo across sizes
        all_names = reader.list_instances(args.sample)
        by_size = {}
        for n in all_names:
            key = "_".join(n.replace("instancia_", "").split("_")[:2])
            by_size.setdefault(key, []).append(n)
        names = []
        for key in sorted(by_size):
            names.extend(sorted(by_size[key])[:2])
        if args.max_instances:
            names = names[:args.max_instances]
    else:
        names = reader.list_instances("small")[:10]

    t0 = time.perf_counter()
    run_sample(names, args.rho, args.runs, args.seed)
    print(f"\n[took {time.perf_counter() - t0:.1f}s]")


if __name__ == "__main__":
    main()
