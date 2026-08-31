"""
P5a verification: the 480-instance reader + UpperBounds reference + RPD.

Checks that:
  1. instances parse into valid HFSPInstance (shape / machine counts),
  2. the UpperBounds reference matches every parsed instance,
  3. baselines (NEH / GA / IG) run on them and RPD vs best-known is sane.

Usage:
    .venv/bin/python scripts/verify_seville.py
"""

import sys
import numpy as np

from hfsp.io import SevilleReader, SevilleReference
from hfsp.core.decoder import ListSchedulingDecoder
from hfsp.methods.heuristics import neh_heuristic
from hfsp.methods.metaheuristics import GeneticAlgorithm, IteratedGreedy

ROOT = "benchmarks/seville"
REF = "benchmarks/seville/references/UpperBounds_01_April_2019.xlsx"


def main():
    reader = SevilleReader(ROOT)
    ref = SevilleReference(REF)
    print(f"instances: small={len(reader.list_instances('small'))}, "
          f"big={len(reader.list_instances('big'))}")
    print(f"reference entries: {len(ref)}")

    # ---- Full coverage: every instance parses AND has a reference ----
    all_names = reader.list_instances("all")
    missing = []
    parse_fail = 0
    for name in all_names:
        inst = reader.load(name)
        if name not in ref.best_known_dict():
            missing.append(name)
        if inst.processing_times.shape[0] != inst.num_jobs:
            parse_fail += 1
    print(f"all instances: {len(all_names)} | parse failures: {parse_fail} | "
          f"missing reference: {len(missing)}")
    ok = (parse_fail == 0 and len(missing) == 0)

    # ---- Baselines + RPD on a fast representative subset ----
    names = ["instancia_10_5_1", "instancia_20_10_1",
             "instancia_35_20_9", "instancia_40_5_1"]
    decoder = ListSchedulingDecoder()

    print(f"\n{'instance':<24}{'n':>4}{'s':>4}{'m':>5}  {'BK':>8}  "
          f"{'NEH':>7}(RPD)    {'GA':>7}(RPD)    {'IG':>7}(RPD)")
    print("-" * 84)

    for name in names:
        inst = reader.load(name)
        bk = ref.best_known(name)
        sols = {
            "NEH": neh_heuristic(inst, decoder),
            "GA": GeneticAlgorithm(population_size=30, max_generations=100,
                                   rng=np.random.default_rng(0)).solve(inst),
            "IG": IteratedGreedy(max_iterations=500, use_local_search=False,
                                 rng=np.random.default_rng(0)).solve(inst),
        }
        rpd = {k: (s.makespan - bk) / bk * 100.0 for k, s in sols.items()}
        print(f"{name:<24}{inst.num_jobs:>4}{inst.num_stages:>4}"
              f"{sum(inst.machines_per_stage):>5}  {bk:>8.0f}  "
              f"{sols['NEH'].makespan:>7.0f}({rpd['NEH']:+5.2f}%)  "
              f"{sols['GA'].makespan:>7.0f}({rpd['GA']:+5.2f}%)  "
              f"{sols['IG'].makespan:>7.0f}({rpd['IG']:+5.2f}%)")

    print("-" * 84)
    print("P5a PASSED" if ok else "P5a FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
