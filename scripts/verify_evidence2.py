"""
Evidence 2: does the anti-collapse evolution actually move the curve down
(and keep the population diverse)?

Runs the FULL new pipeline (LLM mutation light/medium/crossover + Archive +
multi-seed + diversity penalty) on a small instance set, twice:
  - diversity_lambda = 0.0   (baseline, likely collapses)
  - diversity_lambda = 0.05  (anti-collapse)
and prints best-RPD curve + population diversity per generation.

Usage:
    .venv/bin/python scripts/verify_evidence2.py [--gen 10] [--n-runs 3]
"""

import argparse
import json
import time
from pathlib import Path

from hfsp.io import SevilleReader, SevilleReference
from hfsp.llm import (
    make_fitness, EvolutionarySearch, OllamaClient, LLMMutator, ReflectionEngine,
)
from hfsp.llm.archive import InsightArchive
from hfsp.llm.seeds import default_seed_specs

SEVILLE = "benchmarks/seville"
OUT = Path("results/llm/evidence2")

# small instances only -> fast enough to iterate, still shows collapse behavior
SMALL = ["instancia_10_5_1", "instancia_20_10_1", "instancia_35_20_1"]


def run_once(client, fitness, rho, gen, pop, diversity, label):
    mutator = LLMMutator(client, max_retries=2, verbose=True)
    engine = ReflectionEngine(client, verbose=True)
    archive = InsightArchive(client, verbose=False)
    evo = EvolutionarySearch(
        fitness_fn=fitness,
        mutator=mutator.mutate,
        population_size=pop,
        elite_size=2,
        generations=gen,
        seed=42,
        seed_specs=default_seed_specs(),
        use_reflection=True,
        reflection_engine=engine,
        crossover_mutator=mutator.crossover,
        mutation_ratios=(0.3, 0.6, 0.1),
        diversity_lambda=diversity,
        archive=archive,
        n_jobs=6,
        log_mutations=True,
    )
    t0 = time.perf_counter()
    r = evo.run()
    elapsed = time.perf_counter() - t0
    print(f"\n[{label}] diversity={diversity}: best={r['best_fitness']:+.2f}% "
          f"took {elapsed:.0f}s")
    print(f"  best curve: {[f'{h:+.2f}' for h in r['history']]}")
    print(f"  diversity  : {[f'{d:.2f}' for d in r['history_diversity']]}")
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=10)
    parser.add_argument("--pop", type=int, default=8)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    reader = SevilleReader(SEVILLE)
    ref = SevilleReference(f"{SEVILLE}/references/UpperBounds_01_April_2019.xlsx")
    instances = [reader.load(n) for n in SMALL]
    refs = [ref.best_known(n) for n in SMALL]
    budget = max(args.rho * i.num_jobs * i.total_machines / 1000.0 for i in instances)
    fitness = make_fitness(instances, n_runs=args.n_runs, references=refs, seed_base=0,
                           time_limit=budget)

    client = OllamaClient()
    print(f"Evidence 2 | small instances {SMALL} | gen={args.gen} rho={args.rho} "
          f"n_runs={args.n_runs}")

    r0 = run_once(client, fitness, args.rho, args.gen, args.pop, 0.0, "baseline")
    r1 = run_once(client, fitness, args.rho, args.gen, args.pop, 0.05, "anti-collapse")

    summary = {
        "baseline": {"best": r0["best_fitness"], "history": r0["history"],
                     "diversity": r0["history_diversity"]},
        "anti_collapse": {"best": r1["best_fitness"], "history": r1["history"],
                          "diversity": r1["history_diversity"]},
    }
    (OUT / "evidence2.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {OUT / 'evidence2.json'}")
    print("verdict: anti-collapse best should be <= baseline, and its diversity "
          "should not collapse to a flat line.")


if __name__ == "__main__":
    main()
