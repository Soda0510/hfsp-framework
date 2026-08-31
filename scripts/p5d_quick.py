"""
P5d quick reconnaissance: evolved algorithm (g6) vs baselines on the HELD-OUT
test set, under a uniform time budget.

Loads the evolved spec from results/llm/base_run/best_spec.json, runs it and
the main baselines on a stratified sample of the 144-instance test set, and
prints an RPD table + head-to-head g6-vs-IG summary.

Usage:
    .venv/bin/python scripts/p5d_quick.py [--rho 1] [--sizes 10,20,40,80,160]
"""

import argparse
import json
import sys
import time

import numpy as np

from hfsp.io import SevilleReader, SevilleReference
from hfsp.core.decoder import ListSchedulingDecoder
from hfsp.llm import AlgorithmSpec, ConfigurableMetaheuristic
from hfsp.llm.dataset import parse_key
from hfsp.methods.heuristics import neh_heuristic
from hfsp.methods.metaheuristics import GeneticAlgorithm, IteratedGreedy
from hfsp.methods.operators import local_search

SEVILLE = "benchmarks/seville"
BEST_SPEC = "results/llm/base_run/best_spec.json"
TEST_LIST = "benchmarks/seville/split/test.txt"


def load_test_sample(sizes, per_size=2):
    """Stratified sample of the test set across job sizes."""
    reader = SevilleReader(SEVILLE)
    test_names = [ln.strip() for ln in open(TEST_LIST) if ln.strip()]
    chosen = []
    for n in sorted(sizes):
        cands = [nm for nm in test_names if parse_key(nm)[0] == n][:per_size]
        chosen.extend(cands)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--sizes", default="10,20,40,80,160")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    spec = AlgorithmSpec(**json.load(open(BEST_SPEC)))
    print(f"evolved algorithm: {spec.name} (destroy={spec.use_destroy_repair}, "
          f"init={spec.initializer}, accept={spec.acceptance}, ls={spec.use_local_search})")

    names = load_test_sample(sizes)
    reader = SevilleReader(SEVILLE)
    ref = SevilleReference(f"{SEVILLE}/references/UpperBounds_01_April_2019.xlsx")
    print(f"test sample: {len(names)} instances (sizes {sizes})\n")

    decoder = ListSchedulingDecoder()

    def build(algo, rng, tl):
        if algo == "g6":
            return ConfigurableMetaheuristic(spec, rng=rng, time_limit=tl)
        if algo == "IG":
            return IteratedGreedy(max_iterations=50000, use_local_search=True,
                                  rng=rng, time_limit=tl)
        if algo == "GA":
            return GeneticAlgorithm(population_size=60, max_generations=2000,
                                    rng=rng, time_limit=tl)
        raise ValueError(algo)

    algos = ["g6", "IG", "GA"]
    results = {a: [] for a in algos}
    head = {a: {"win": 0, "loss": 0, "tie": 0} for a in algos}

    t0 = time.perf_counter()
    for name in names:
        inst = reader.load(name)
        bk = ref.best_known(name)
        tl = args.rho * inst.num_jobs * inst.total_machines / 1000.0
        scores = {}
        for a in algos:
            rng = np.random.default_rng(42)
            scores[a] = build(a, rng, tl).solve(inst).makespan
        for a in algos:
            results[a].append((scores[a] - bk) / bk * 100.0)
        # head-to-head g6 vs each other algo
        for a in algos:
            if a == "g6":
                continue
            if scores["g6"] < scores[a] - 1e-9:
                head[a]["win"] += 1
            elif scores["g6"] > scores[a] + 1e-9:
                head[a]["loss"] += 1
            else:
                head[a]["tie"] += 1

    print(f"{'algorithm':<6}{'mean RPD':>10}{'min':>9}{'max':>9}")
    print("-" * 36)
    for a in algos:
        v = results[a]
        print(f"{a:<6}{np.mean(v):>10.2f}{np.min(v):>9.2f}{np.max(v):>9.2f}")

    print(f"\nhead-to-head g6 (win/loss/tie vs each):")
    for a in ["IG", "GA"]:
        h = head[a]
        print(f"  g6 vs {a:<3}: {h['win']}W / {h['loss']}L / {h['tie']}T")
    print(f"\n[took {time.perf_counter() - t0:.0f}s]")


if __name__ == "__main__":
    main()
