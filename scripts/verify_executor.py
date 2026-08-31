"""
P1 verification: ConfigurableMetaheuristic reproduces existing algorithms.

Runs each seed AlgorithmSpec through BOTH the executor and the original
hand-crafted implementation with the same RNG seed, then compares makespans:

    GA spec      == GeneticAlgorithm    (exact)
    IG spec      == IteratedGreedy      (exact)
    NEH+LS spec  == NEH + local_search  (exact)
    SA-like spec ~= SimulatedAnnealing  (comparable, NOT bit-identical: the
                    executor uses a simplified single-temperature loop)

Usage:
    python scripts/verify_executor.py [instance_name]
"""

import sys
import argparse

import numpy as np

from hfsp.io import InstanceReader
from hfsp.core.decoder import ListSchedulingDecoder
from hfsp.llm import AlgorithmSpec, ConfigurableMetaheuristic
from hfsp.llm.seeds import ga_seed_spec, ig_seed_spec, sa_seed_spec, neh_ls_seed_spec
from hfsp.methods.heuristics import neh_heuristic
from hfsp.methods.metaheuristics import GeneticAlgorithm, IteratedGreedy, SimulatedAnnealing
from hfsp.methods.operators import local_search


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def check(instance_name: str) -> bool:
    reader = InstanceReader("Data")
    instance = reader.load(instance_name)
    decoder = ListSchedulingDecoder()
    ok = True

    print(f"\n{'Algorithm':<10}{'original':>12}{'executor':>12}  result")
    print("-" * 52)

    # 1) GA: exact match
    seed = 42
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    ga_orig = GeneticAlgorithm(rng=rng1).solve(instance).makespan
    ga_spec = ga_seed_spec()
    ga_exec = ConfigurableMetaheuristic(ga_spec, rng=rng2).solve(instance).makespan
    exact = abs(ga_orig - ga_exec) < 1e-9
    ok &= exact
    print(f"{'GA':<10}{ga_orig:>12.2f}{ga_exec:>12.2f}  {'EXACT' if exact else 'MISMATCH!'}")

    # 2) IG: exact match
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    ig_orig = IteratedGreedy(rng=rng1).solve(instance).makespan
    ig_spec = ig_seed_spec()
    ig_exec = ConfigurableMetaheuristic(ig_spec, rng=rng2).solve(instance).makespan
    exact = abs(ig_orig - ig_exec) < 1e-9
    ok &= exact
    print(f"{'IG':<10}{ig_orig:>12.2f}{ig_exec:>12.2f}  {'EXACT' if exact else 'MISMATCH!'}")

    # 3) NEH+LS: exact match vs neh_heuristic + local_search
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    ref = local_search(neh_heuristic(instance, decoder, rng1), decoder,
                       max_iterations=100, strategy="first_improvement", rng=rng1)
    neh_ls_spec = neh_ls_seed_spec()
    exec_sol = ConfigurableMetaheuristic(neh_ls_spec, rng=rng2).solve(instance)
    exact = abs(ref.makespan - exec_sol.makespan) < 1e-9
    ok &= exact
    print(f"{'NEH+LS':<10}{ref.makespan:>12.2f}{exec_sol.makespan:>12.2f}  "
          f"{'EXACT' if exact else 'MISMATCH!'}")

    # 4) SA-like: comparable (different loop structure, so not exact)
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    sa_orig = SimulatedAnnealing(
        initial_temperature=100.0, cooling_rate=0.97,
        max_iterations=100, max_total_iterations=10000, rng=rng1,
    ).solve(instance).makespan
    sa_spec = sa_seed_spec()
    sa_exec = ConfigurableMetaheuristic(sa_spec, rng=rng2).solve(instance).makespan
    diff = abs(sa_orig - sa_exec) / max(sa_orig, 1e-9) * 100
    close = diff < 15.0
    ok &= close
    print(f"{'SA-like':<10}{sa_orig:>12.2f}{sa_exec:>12.2f}  "
          f"{'close' if close else 'DIFFERENT!'} ({diff:.1f}%)")

    # 5) Constructive: SPT / LPT quick sanity (executor runs, no crash)
    for init in ("spt", "lpt"):
        rng = np.random.default_rng(seed)
        spec = AlgorithmSpec(name=init.upper(), initializer=init, max_iterations=0,
                             use_local_search=False)
        sol = ConfigurableMetaheuristic(spec, rng=rng).solve(instance)
        print(f"{init.upper():<10}{sol.makespan:>12.2f}{'':>12}  runs-ok")

    print("-" * 52)
    if ok:
        print("ALL CHECKS PASSED: executor reproduces hand-crafted algorithms.")
    else:
        print("SOME CHECKS FAILED — see above.")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", nargs="?", default="10-3-3")
    args = parser.parse_args()
    ok = check(args.instance)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
