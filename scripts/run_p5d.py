"""
Formal P5d: final evolved algorithm vs baselines on the 144-instance HELD-OUT
test set, under a uniform time budget, with statistical tests.

Runs the evolved spec (results/llm/base_run/best_spec.json) and the baseline
metaheuristics on every test instance once (same seed per instance), computes
RPD vs the published UpperBounds, and reports:
  - mean/min/max RPD per algorithm,
  - head-to-head wins of the evolved algorithm vs each baseline,
  - Friedman + Wilcoxon signed-rank tests.

Outputs (results/llm/p5d/):
    rpd_table.csv        algorithm, instance, rpd
    summary.json         mean RPD, wins, p-values

Usage:
    .venv/bin/python scripts/run_p5d.py [--rho 2] [--runs 1] [--limit N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hfsp.io import SevilleReader, SevilleReference
from hfsp.core.decoder import ListSchedulingDecoder
from hfsp.llm import AlgorithmSpec, ConfigurableMetaheuristic
from hfsp.methods.heuristics import neh_heuristic
from hfsp.methods.metaheuristics import (
    GeneticAlgorithm, SimulatedAnnealing, IteratedGreedy, DiscretePSO,
)
from hfsp.methods.operators import local_search

SEVILLE = "benchmarks/seville"
BEST_SPEC = "results/llm/base_run/best_spec.json"
TEST_LIST = "benchmarks/seville/split/test.txt"
OUT_DIR = Path("results/llm/p5d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="cap test instances")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = AlgorithmSpec(**json.load(open(BEST_SPEC)))
    reader = SevilleReader(SEVILLE)
    ref = SevilleReference(f"{SEVILLE}/references/UpperBounds_01_April_2019.xlsx")
    names = [ln.strip() for ln in open(TEST_LIST) if ln.strip()]
    if args.limit:
        names = names[:args.limit]

    decoder = ListSchedulingDecoder()

    def build(algo, rng, tl):
        if algo == "Evolved(g1)":
            return ConfigurableMetaheuristic(spec, rng=rng, time_limit=tl)
        if algo == "IG":
            return IteratedGreedy(max_iterations=50000, use_local_search=True,
                                  rng=rng, time_limit=tl)
        if algo == "GA":
            return GeneticAlgorithm(population_size=60, max_generations=2000,
                                    rng=rng, time_limit=tl)
        if algo == "SA":
            return SimulatedAnnealing(initial_temperature=100.0, cooling_rate=0.97,
                                      max_iterations=200, max_total_iterations=500000,
                                      rng=rng, time_limit=tl)
        if algo == "DPSO":
            return DiscretePSO(swarm_size=40, max_iterations=2000, w=0.5, c1=0.3, c2=0.2,
                               rng=rng, time_limit=tl)
        if algo == "NEH+LS":
            def solve(inst):
                sol = neh_heuristic(inst, decoder, rng)
                return local_search(sol, decoder, max_iterations=200,
                                    strategy="first_improvement", rng=rng)
            from types import SimpleNamespace
            return SimpleNamespace(solve=solve)
        if algo == "NEH":
            from types import SimpleNamespace
            return SimpleNamespace(solve=lambda inst: neh_heuristic(inst, decoder, rng))
        raise ValueError(algo)

    algos = ["Evolved(g1)", "IG", "GA", "SA", "DPSO", "NEH+LS", "NEH"]
    rows = []          # (algorithm, instance, rpd)
    per_algo = {a: [] for a in algos}

    t0 = time.perf_counter()
    for name in names:
        inst = reader.load(name)
        bk = ref.best_known(name)
        tl = args.rho * inst.num_jobs * inst.total_machines / 1000.0
        for r in range(args.runs):
            scores = {}
            for a in algos:
                rng = np.random.default_rng(42 + r)
                scores[a] = build(a, rng, tl).solve(inst).makespan
            for a in algos:
                rpd = (scores[a] - bk) / bk * 100.0
                per_algo[a].append(rpd)
                rows.append((a, name, rpd))

    df = pd.DataFrame(rows, columns=["algorithm", "instance", "rpd"])
    df.to_csv(OUT_DIR / "rpd_table.csv", index=False)

    print(f"test instances: {len(names)} | rho={args.rho} | runs={args.runs} | "
          f"took {time.perf_counter()-t0:.0f}s\n")
    print(f"{'algorithm':<12}{'mean RPD':>10}{'min':>9}{'max':>9}")
    print("-" * 42)
    summary = {}
    for a in algos:
        v = np.array(per_algo[a])
        summary[a] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
        print(f"{a:<12}{np.mean(v):>10.2f}{np.min(v):>9.2f}{np.max(v):>9.2f}")

    # Head-to-head + Wilcoxon (paired) + Friedman
    evo = "Evolved(g1)"
    summary["head_to_head"] = {}
    summary["wilcoxon"] = {}
    print(f"\nhead-to-head {evo} (win/loss/tie | Wilcoxon p) vs each:")
    for a in algos:
        if a == evo:
            continue
        e = np.array(per_algo[evo]); o = np.array(per_algo[a])
        w = int(np.sum(e < o - 1e-9)); l = int(np.sum(e > o + 1e-9)); t = int(np.sum(np.abs(e-o) <= 1e-9))
        p = stats.wilcoxon(e, o, zero_method="wilcox").pvalue if len(e) > 0 else 1.0
        summary["head_to_head"][a] = {"win": w, "loss": l, "tie": t}
        summary["wilcoxon"][a] = float(p)
        print(f"  vs {a:<8}: {w:>3}W/{l:>3}L/{t:>3}T   p={p:.4f}")

    # Friedman across all algorithms (per-instance ranks)
    try:
        mat = np.column_stack([per_algo[a] for a in algos])
        f_stat, f_p = stats.friedmanchisquare(*[mat[:, i] for i in range(mat.shape[1])])
        summary["friedman"] = {"stat": float(f_stat), "p": float(f_p)}
        print(f"\nFriedman: stat={f_stat:.2f}, p={f_p:.4f}")
    except Exception as ex:
        summary["friedman"] = {"error": str(ex)}
        print(f"\nFriedman failed: {ex}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved: {OUT_DIR / 'rpd_table.csv'} + {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
