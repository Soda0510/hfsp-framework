"""
Evaluate a given evolved spec on the 144-instance HELD-OUT test set, with
live progress + timing + ETA.  Baselines (IG/GA/...) are reused from the
existing P5d rpd_table.csv, so only THIS spec needs to be run.

Usage:
    .venv/bin/python scripts/eval_spec_test.py \
        --spec results/llm/ablation_random/best_spec.json --label g10 --rho 2
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hfsp.io import SevilleReader, SevilleReference
from hfsp.llm import AlgorithmSpec, ConfigurableMetaheuristic

SEVILLE = "benchmarks/seville"
TEST_LIST = "benchmarks/seville/split/test.txt"
P5D_CSV = "results/llm/p5d/rpd_table.csv"
OUT_DIR = Path("results/llm/test_evals")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="path to best_spec.json")
    parser.add_argument("--label", required=True, help="e.g. g10, oneshot")
    parser.add_argument("--rho", type=float, default=2.0)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = AlgorithmSpec(**json.load(open(args.spec)))
    print(f"evaluating spec '{args.label}': "
          f"pop={spec.population_mode} destroy={spec.use_destroy_repair} "
          f"init={spec.initializer} accept={spec.acceptance} ls={spec.use_local_search}")

    reader = SevilleReader(SEVILLE)
    ref = SevilleReference(f"{SEVILLE}/references/UpperBounds_01_April_2019.xlsx")
    names = [ln.strip() for ln in open(TEST_LIST) if ln.strip()]

    rows = []
    t0 = time.perf_counter()
    for i, name in enumerate(names):
        inst = reader.load(name)
        bk = ref.best_known(name)
        tl = args.rho * inst.num_jobs * inst.total_machines / 1000.0
        rng = np.random.default_rng(42)          # same seed as P5d
        sol = ConfigurableMetaheuristic(spec, rng=rng, time_limit=tl).solve(inst)
        rpd = (sol.makespan - bk) / bk * 100.0
        rows.append((args.label, name, rpd))

        if (i + 1) % 15 == 0 or (i + 1) == len(names):
            elapsed = time.perf_counter() - t0
            per = elapsed / (i + 1)
            eta = per * (len(names) - (i + 1))
            print(f"  [{elapsed:>5.0f}s] {i + 1}/{len(names)} instances "
                  f"({per:.1f}s/inst, ETA {eta:.0f}s)")

    out_csv = OUT_DIR / f"{args.label}_rpd.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "instance", "rpd"])
        w.writerows(rows)
    print(f"saved -> {out_csv}")

    # ---- Compare with P5d baselines (existing data) ----
    p5d = pd.read_csv(P5D_CSV)
    mine = pd.read_csv(out_csv)
    merged = pd.concat([p5d, mine], ignore_index=True)
    piv = merged.pivot(index="instance", columns="algorithm", values="rpd")

    print(f"\nmean RPD comparison (test set):")
    print(f"{'algorithm':<16}{'mean':>9}{'min':>8}")
    print("-" * 34)
    for a in piv.columns:
        print(f"{a:<16}{piv[a].mean():>9.2f}{piv[a].min():>8.2f}")

    evo, ig = args.label, "IG"
    if evo in piv and ig in piv:
        e, o = piv[evo], piv[ig]
        w = int((e < o - 1e-9).sum()); l = int((e > o + 1e-9).sum()); t = int((abs(e - o) <= 1e-9).sum())
        print(f"\n{evo} vs {ig}: {w}W/{l}L/{t}T (of {len(piv)})")
    if args.label in piv and "Evolved(g1)" in piv:
        e, o = piv[args.label], piv["Evolved(g1)"]
        w = int((e < o - 1e-9).sum()); l = int((e > o + 1e-9).sum()); t = int((abs(e - o) <= 1e-9).sum())
        print(f"{args.label} vs Evolved(g1): {w}W/{l}L/{t}T (of {len(piv)})")


if __name__ == "__main__":
    main()
