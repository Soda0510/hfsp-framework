"""
P3 verification: LLM-guided evolution with the local model.

Part 1 (fast, 10-job): the core hypothesis — does an LLM mutation IMPROVE a
mediocre parent?  Mutate the two worst seeds (GA, NEH+LS) and compare child
vs parent fitness.

Part 2 (heavier, 20-job): a short full evolutionary run driven by the LLM
mutator, compared against the best seed and the random-mutation baseline.

Usage:
    .venv/bin/python scripts/run_llm_evolution.py [--llm-gen 2] [--llm-pop 6]
"""

import argparse
import sys
import time

import numpy as np

from hfsp.io import InstanceReader
from hfsp.llm import (
    make_fitness, per_instance_best, EvolutionarySearch,
    random_spec, perturb_spec, OllamaClient, LLMMutator,
)
from hfsp.llm.seeds import default_seed_specs, ga_seed_spec, neh_ls_seed_spec


def evaluate_pair(fitness, parent, child, label):
    """Evaluate a parent/child pair; report whether the child improved."""
    fp = fitness(parent)
    fc = fitness(child)
    improved = fc < fp - 1e-9
    print(f"  {label:>10}: parent {fp:+.2f}% -> child {fc:+.2f}%  "
          f"{'IMPROVED' if improved else 'no'}")
    return improved


def part1_improve_parent(instances, fitness):
    """Core P3 check: LLM mutation improves mediocre parents."""
    print("\n[P3 Part 1] Does LLM mutation improve a mediocre parent?")
    client = OllamaClient()
    mut = LLMMutator(client, max_retries=2, verbose=False)
    parents = [ga_seed_spec(), neh_ls_seed_spec()]
    wins = 0
    for p in parents:
        for i in range(2):  # two independent mutations per parent
            child = mut.mutate(p, None, f"{p.name}-llm{i}", {"fitness": fitness(p)})
            if evaluate_pair(fitness, p, child, f"{p.name}#{i}"):
                wins += 1
    print(f"  improvements: {wins}/{len(parents) * 2}")
    return wins > 0


def part2_llm_evolution(instances, refs, seed_specs, args):
    """Short full LLM-driven evolution on 20-job instances."""
    print("\n[P3 Part 2] LLM-driven evolution (20-job, "
          f"pop={args.llm_pop}, gen={args.llm_gen})")
    TIME_BUDGET = 2.0
    fitness = make_fitness(instances, n_runs=1, references=refs, seed_base=0,
                           time_limit=TIME_BUDGET)

    client = OllamaClient()
    llm_mut = LLMMutator(client, max_retries=2, verbose=True)

    evo = EvolutionarySearch(
        fitness_fn=fitness,
        mutator=llm_mut.mutate,
        population_size=args.llm_pop,
        elite_size=2,
        generations=args.llm_gen,
        seed=args.seed,
        seed_specs=seed_specs,
    )
    t0 = time.perf_counter()
    result = evo.run()
    elapsed = time.perf_counter() - t0
    print(f"\n  LLM evolution took {elapsed:.1f}s; mutator stats: {llm_mut.stats}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-gen", type=int, default=2)
    parser.add_argument("--llm-pop", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    reader = InstanceReader("Data")
    seeds = default_seed_specs()

    # ---- Part 1: fast parent-improvement check on 10-job ----
    small = [reader.load(n) for n in ["10-3-3", "10-3-4", "10-3-6"]]
    refs_small = per_instance_best(small, seeds, n_runs=1, seed_base=0, time_limit=1.0)
    fit_small = make_fitness(small, n_runs=1, references=refs_small, seed_base=0,
                             time_limit=1.0)
    ok = part1_improve_parent(small, fit_small)

    # ---- Part 2: short LLM evolution on 20-job (where seeds do NOT saturate) ----
    big = [reader.load(n) for n in ["20-3-3", "20-3-4"]]
    refs_big = per_instance_best(big, seeds, n_runs=1, seed_base=0, time_limit=2.0)
    print(f"\n[P3 Part 2] 20-job refs (best of seeds): {refs_big}")
    res = part2_llm_evolution(big, refs_big, seeds, args)
    print(f"  best spec: {res['best_spec'].name}  RPD {res['best_fitness']:+.2f}%")
    print(f"  convergence: {[f'{h:+.2f}' for h in res['history']]}")

    # Compare vs random-mutation baseline (quick, same budget).
    rng = np.random.default_rng(args.seed)
    random_fits = [make_fitness(big, n_runs=1, references=refs_big, seed_base=0,
                                time_limit=2.0)(random_spec(rng, "b")) for _ in range(4)]
    random_mean = float(np.mean(random_fits))
    print(f"  random-mutation baseline (4 random specs): mean RPD {random_mean:+.2f}%")

    if ok:
        print("\nP3 PART 1 PASSED: LLM mutation improves parents.")
    else:
        print("\nP3 PART 1 FAILED: no LLM mutation improved its parent.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
