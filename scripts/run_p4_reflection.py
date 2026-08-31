"""
P4 verification: reflection module.

Part 1 (real LLM): generate a reflection for one spec and print the critique.
Part 2 (real LLM): a tiny evolutionary run with LLM mutation + reflection,
                   confirming the reflection text reaches the mutation prompt
                   and that mutation logging (M8 data) is recorded.
Part 3 (fast, no LLM): dummy reflection + random mutation — confirms the
                   ablation toggle and logging mechanics.

Usage:
    .venv/bin/python scripts/run_p4_reflection.py [--gen 1] [--pop 4]
"""

import argparse
import sys

import numpy as np

from hfsp.io import InstanceReader
from hfsp.llm import (
    make_fitness, per_instance_best, EvolutionarySearch,
    perturb_spec, OllamaClient, LLMMutator, ReflectionEngine, DummyReflectionEngine,
)
from hfsp.llm.seeds import default_seed_specs, neh_ls_seed_spec


def part1_real_reflection():
    """Generate and print one real critique from the local LLM."""
    print("[P4 Part 1] Real reflection on a mediocre spec")
    client = OllamaClient()
    engine = ReflectionEngine(client, verbose=False)
    spec = neh_ls_seed_spec()   # a mediocre performer (constructive only)
    critique = engine.get_or_reflect(spec, 3.8)
    print(f"  spec: {spec.name}")
    print(f"  critique: {critique}")
    print(f"  stats: {engine.stats}")
    return bool(critique)


def part2_real_evolution(args):
    """Tiny evolution with real LLM mutation + reflection + logging."""
    print("\n[P4 Part 2] LLM mutation + reflection + mutation log "
          f"(pop={args.pop}, gen={args.gen})")
    reader = InstanceReader("Data")
    instances = [reader.load(n) for n in ["10-3-3", "10-3-4"]]
    seeds = default_seed_specs()
    refs = per_instance_best(instances, seeds, n_runs=1, seed_base=0, time_limit=1.0)
    fitness = make_fitness(instances, n_runs=1, references=refs, seed_base=0,
                           time_limit=1.0)

    client = OllamaClient()
    mutator = LLMMutator(client, max_retries=2, verbose=True)
    engine = ReflectionEngine(client, verbose=True)

    evo = EvolutionarySearch(
        fitness_fn=fitness,
        mutator=mutator.mutate,
        population_size=args.pop,
        elite_size=2,
        generations=args.gen,
        seed=args.seed,
        seed_specs=seeds,
        use_reflection=True,
        reflection_engine=engine,
        log_mutations=True,
    )
    result = evo.run()

    print(f"  best: {result['best_spec'].name} RPD {result['best_fitness']:+.2f}%")
    print(f"  reflection stats: {engine.stats}")
    print(f"  mutations logged: {len(evo.mutation_log)}")
    if evo.mutation_log:
        sample = evo.mutation_log[0]
        print(f"  sample log: gen={sample['gen']} "
              f"parent={sample['parent']['name']} fit={sample['parent_fitness']:.2f} "
              f"child={sample['child']['name']}")
    return True


def part3_fast_dummy():
    """No-LLM sanity: dummy reflection + random mutation, logging on."""
    print("\n[P4 Part 3] Dummy reflection + random mutation (no LLM)")
    reader = InstanceReader("Data")
    instances = [reader.load(n) for n in ["10-3-3"]]
    seeds = default_seed_specs()
    refs = per_instance_best(instances, seeds, n_runs=1, seed_base=0, time_limit=1.0)
    fitness = make_fitness(instances, n_runs=1, references=refs, seed_base=0,
                           time_limit=1.0)

    engine = DummyReflectionEngine()
    evo = EvolutionarySearch(
        fitness_fn=fitness,
        mutator=perturb_spec,          # random mutation
        population_size=4,
        elite_size=2,
        generations=1,
        seed=0,
        seed_specs=seeds,
        use_reflection=True,
        reflection_engine=engine,
        log_mutations=True,
    )
    result = evo.run()
    print(f"  best: {result['best_spec'].name} RPD {result['best_fitness']:+.2f}%")
    print(f"  mutations logged: {len(evo.mutation_log)}")
    print("  P4 part 3 OK")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=1)
    parser.add_argument("--pop", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    ok1 = part1_real_reflection()
    ok2 = part2_real_evolution(args)
    ok3 = part3_fast_dummy()

    print("\n" + ("P4 PASSED" if (ok1 and ok2 and ok3) else "P4 FAILED"))
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


if __name__ == "__main__":
    main()
