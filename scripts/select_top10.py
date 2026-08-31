"""
Select the TOP-K algorithm specs seen during evolution (E: portfolio).

Collects every spec that appeared in the base evolution (seeds + all mutated
children), re-evaluates each with the SAME fitness as evolution (training
subset, rho=0.3, UpperBounds reference), ranks them, and reports the top-K with
a structural summary — so we can judge whether they are diverse or near-clones.

Usage:
    .venv/bin/python scripts/select_top10.py [--k 10] [--rho 0.3]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from hfsp.io import SevilleReader, SevilleReference
from hfsp.llm import make_fitness, spec_from_dict
from hfsp.llm.seeds import default_seed_specs

SEVILLE = "benchmarks/seville"
REF_XLSX = "benchmarks/seville/references/UpperBounds_01_April_2019.xlsx"
SPLIT_SUBSET = "benchmarks/seville/split/evolution_train_subset.txt"
MUT_LOG = "results/llm/base_run/mutation_log.jsonl"
OUT = Path("results/llm/e_top10")


def _sig(spec):
    d = spec.to_dict()
    d.pop("name", None)
    return json.dumps(d, sort_keys=True, default=str)


def structural_summary(spec):
    s = spec
    return {
        "pop": int(s.population_mode),
        "destroy": int(s.use_destroy_repair),
        "init": s.initializer,
        "accept": s.acceptance,
        "ls": int(s.use_local_search),
        "mut_ops": ",".join(s.mutation_operators),
        "max_iter": s.max_iterations,
        "pop_size": s.population_size,
        "temp": s.temperature,
        "cool": s.cooling_rate,
        "d_size": s.destruction_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Collect unique specs: seeds + all children from the mutation log
    specs = []
    seen = set()
    for s in default_seed_specs():
        if _sig(s) not in seen:
            seen.add(_sig(s))
            specs.append(s)
    for line in open(MUT_LOG):
        child = json.loads(line)["child"]
        try:
            s = spec_from_dict(child, name=child.get("name", "cand"))
        except Exception:
            continue
        if _sig(s) not in seen:
            seen.add(_sig(s))
            specs.append(s)
    print(f"unique specs to evaluate: {len(specs)}")

    # Same fitness as evolution
    train_names = [ln.strip() for ln in open(SPLIT_SUBSET) if ln.strip()]
    reader = SevilleReader(SEVILLE)
    ref = SevilleReference(REF_XLSX)
    instances = [reader.load(n) for n in train_names]
    refs = [ref.best_known(n) for n in train_names]
    budget = max(args.rho * i.num_jobs * i.total_machines / 1000.0 for i in instances)
    fitness = make_fitness(instances, n_runs=1, references=refs, seed_base=0,
                           time_limit=budget)

    ranked = []
    t0 = time.perf_counter()
    for i, s in enumerate(specs):
        ranked.append((fitness(s), s))
        if (i + 1) % 25 == 0 or (i + 1) == len(specs):
            elapsed = time.perf_counter() - t0
            per = elapsed / (i + 1)
            eta = per * (len(specs) - (i + 1))
            print(f"  [{elapsed:.0f}s] evaluated {i + 1}/{len(specs)} "
                  f"({per:.1f}s/spec, ETA {eta:.0f}s)")
    ranked.sort(key=lambda t: t[0])

    print(f"\nTOP-{args.k} specs (mean RPD on training subset):")
    print(f"{'#':>2} {'name':<12}{'RPD':>8}  pop destroy init     accept   ls  mut_ops          max_iter  pop_sz  temp   d_size")
    print("-" * 100)
    rows = []
    for idx, (f, s) in enumerate(ranked[:args.k]):
        st = structural_summary(s)
        print(f"{idx+1:>2} {s.name:<12}{f:>+8.2f}  "
              f"{st['pop']:>3} {st['destroy']:>6}  {st['init']:<5}   {st['accept']:<9}  "
              f"{st['ls']:>1}  {st['mut_ops']:<14} {st['max_iter']:>8}  {st['pop_size']:>6}  "
              f"{str(st['temp']):>5}  {st['d_size']}")
        rows.append({"rank": idx + 1, "name": s.name, "rpd": round(f, 4), **st})
    with open(OUT / "top10.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {OUT / 'top10.csv'}")


if __name__ == "__main__":
    main()
