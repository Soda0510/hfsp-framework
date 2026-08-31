"""
P5c: base-model LLM evolution (zero-shot qwen2.5) on the training set.

This run is the BASE-LINE of the fine-tuning comparison, and it ALSO collects
the mutation log (parent/child specs + fitness) that becomes the fine-tuning
training data (M8).

Outputs (saved to results/llm/base_run/):
    best_spec.json        the final evolved algorithm (AlgorithmSpec)
    convergence.json      best RPD per generation
    mutation_log.jsonl    (parent, child, parent_fitness) per mutation

Usage:
    .venv/bin/python scripts/run_evolution_full.py [--gen 8] [--pop 8] [--rho 2]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from hfsp.io import SevilleReader, SevilleReference
from hfsp.llm import (
    make_fitness, EvolutionarySearch,
    OllamaClient, LLMMutator, ReflectionEngine,
)
from hfsp.llm.archive import InsightArchive
from hfsp.llm.seeds import default_seed_specs

OUT_DIR = Path("results/llm/base_run")
SEVILLE_ROOT = "benchmarks/seville"
REF_XLSX = "benchmarks/seville/references/UpperBounds_01_April_2019.xlsx"
SPLIT_SUBSET = "benchmarks/seville/split/evolution_train_subset.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=10)
    parser.add_argument("--pop", type=int, default=8)
    parser.add_argument("--rho", type=float, default=0.5,
                        help="time budget factor for evolution evals (larger = better ranking)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-reflection", action="store_true", default=True)
    parser.add_argument("--jobs", type=int, default=6,
                        help="parallel evaluation workers")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="multi-seed evaluation runs per instance (阶段1, 降噪声)")
    parser.add_argument("--diversity", type=float, default=0.0,
                        help="diversity penalty lambda (阶段4, >0 enables anti-collapse)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Training set: stratified subset of the 480 (see dataset.py), sized to
    # span the distribution while keeping evolution evaluation fast.
    train_names = [ln.strip() for ln in open(SPLIT_SUBSET) if ln.strip()]
    reader = SevilleReader(SEVILLE_ROOT)
    ref = SevilleReference(REF_XLSX)
    instances = [reader.load(n) for n in train_names]
    seeds = default_seed_specs()
    print(f"training instances: {len(train_names)} (from 480 stratified split)")

    budget = lambda inst: args.rho * inst.num_jobs * inst.total_machines / 1000.0
    # Reference = published UpperBounds (absolute target, not seed-min).
    refs = [ref.best_known(n) for n in train_names]
    print(f"fitness reference: published UpperBounds (mean {np.mean(refs):.0f})")

    fitness = make_fitness(instances, n_runs=args.n_runs, references=refs, seed_base=0,
                           time_limit=max(budget(i) for i in instances))

    client = OllamaClient()
    mutator = LLMMutator(client, max_retries=2, verbose=True)
    engine = ReflectionEngine(client, verbose=True)
    archive = InsightArchive(client, verbose=True)

    evo = EvolutionarySearch(
        fitness_fn=fitness,
        mutator=mutator.mutate,
        population_size=args.pop,
        elite_size=2,
        generations=args.gen,
        seed=args.seed,
        seed_specs=seeds,
        use_reflection=args.use_reflection,
        reflection_engine=engine,
        log_mutations=True,
        n_jobs=args.jobs,
        crossover_mutator=mutator.crossover,
        mutation_ratios=(0.3, 0.6, 0.1),   # 轻/中/交叉
        diversity_lambda=args.diversity,
        archive=archive,
    )

    t0 = time.perf_counter()
    print(f"\n=== BASE-MODEL evolution (pop={args.pop}, gen={args.gen}) ===")
    result = evo.run()
    elapsed = time.perf_counter() - t0

    # ---- Save outputs (JSON + plot-ready CSV) ----
    import csv
    (OUT_DIR / "best_spec.json").write_text(
        json.dumps(result["best_spec"].to_dict(), indent=2))
    (OUT_DIR / "convergence.json").write_text(
        json.dumps({"history": result["history"],
                    "history_mean": result["history_mean"]}, indent=2))
    with open(OUT_DIR / "mutation_log.jsonl", "w") as f:
        for entry in evo.mutation_log:
            f.write(json.dumps(entry) + "\n")

    # evolution_curve.csv: gen, best RPD, population-mean RPD  (for plotting)
    with open(OUT_DIR / "evolution_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_rpd", "mean_rpd", "diversity"])
        for i, (b, m, d) in enumerate(zip(result["history"], result["history_mean"],
                                          result["history_diversity"])):
            w.writerow([i, f"{b:.4f}", f"{m:.4f}", f"{d:.3f}"])

    # seed_baselines.csv: each seed's RPD (reference lines on the plot)
    with open(OUT_DIR / "seed_baselines.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "rpd"])
        for s in seeds:
            w.writerow([s.name, f"{fitness(s):.4f}"])

    print(f"\n{'='*60}")
    print(f"BEST evolved spec: {result['best_spec'].name}  RPD {result['best_fitness']:+.2f}%")
    print(f"best curve: {[f'{h:+.2f}' for h in result['history']]}")
    print(f"mean curve: {[f'{m:+.2f}' for m in result['history_mean']]}")
    print(f"elapsed: {elapsed:.0f}s | LLM mutator stats: {mutator.stats}")
    print(f"reflection stats: {engine.stats}")
    print(f"saved: best_spec.json / convergence.json / evolution_curve.csv / "
          f"seed_baselines.csv / mutation_log.jsonl -> {OUT_DIR}")


if __name__ == "__main__":
    main()
