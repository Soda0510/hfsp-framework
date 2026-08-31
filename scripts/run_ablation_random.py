"""
Ablation level 0: random-mutation evolution (NO LLM prior).

Same training set / fitness reference / generations as the zero-shot LLM run
(P5c), but the mutation operator is RANDOM (perturb_spec) instead of the LLM.
This isolates the contribution of the LLM's pretrained prior:

    level 0 (random)      vs      level 1 (zero-shot LLM)
    [no prior, no learn]          [pretrained prior, no post-training]

Outputs -> results/llm/ablation_random/:
    best_spec.json / evolution_curve.csv / seed_baselines.csv / mutation_log.jsonl

Usage:
    .venv/bin/python scripts/run_ablation_random.py [--gen 20] [--pop 8] [--rho 0.3]
"""

import argparse
import csv
import json
from pathlib import Path

from hfsp.io import SevilleReader, SevilleReference
from hfsp.llm import make_fitness, EvolutionarySearch, perturb_spec
from hfsp.llm.seeds import default_seed_specs

SEVILLE = "benchmarks/seville"
REF_XLSX = "benchmarks/seville/references/UpperBounds_01_April_2019.xlsx"
SPLIT_SUBSET = "benchmarks/seville/split/evolution_train_subset.txt"
OUT_DIR = Path("results/llm/ablation_random")
BASE_RUN_DIR = Path("results/llm/base_run")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=10)
    parser.add_argument("--pop", type=int, default=8)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--n-runs", type=int, default=3,
                        help="multi-seed evaluation runs per instance (阶段1)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_names = [ln.strip() for ln in open(SPLIT_SUBSET) if ln.strip()]
    reader = SevilleReader(SEVILLE)
    ref = SevilleReference(REF_XLSX)
    instances = [reader.load(n) for n in train_names]
    refs = [ref.best_known(n) for n in train_names]
    budget = max(args.rho * i.num_jobs * i.total_machines / 1000.0 for i in instances)
    fitness = make_fitness(instances, n_runs=args.n_runs, references=refs, seed_base=0,
                           time_limit=budget)

    print(f"ablation level 0 (random mutation): {len(train_names)} train instances, "
          f"gen={args.gen}, pop={args.pop}, rho={args.rho}")

    evo = EvolutionarySearch(
        fitness_fn=fitness,
        mutator=perturb_spec,          # RANDOM mutation (no LLM)
        population_size=args.pop,
        elite_size=2,
        generations=args.gen,
        seed=args.seed,
        seed_specs=default_seed_specs(),
        use_reflection=False,
        log_mutations=True,
        n_jobs=args.jobs,
    )
    result = evo.run()

    # ---- Save ----
    (OUT_DIR / "best_spec.json").write_text(
        json.dumps(result["best_spec"].to_dict(), indent=2))
    with open(OUT_DIR / "evolution_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_rpd", "mean_rpd"])
        for i, (b, m) in enumerate(zip(result["history"], result["history_mean"])):
            w.writerow([i, f"{b:.4f}", f"{m:.4f}"])

    # ---- Comparison with level 1 (zero-shot LLM) ----
    base_conv = json.load(open(BASE_RUN_DIR / "convergence.json"))
    llm_best = base_conv["history"][-1]          # g1's best RPD
    random_best = result["best_fitness"]

    print("\n" + "=" * 50)
    print(f"level 0 (random)       best RPD: {random_best:+.2f}%   ({result['best_spec'].name})")
    print(f"level 1 (zero-shot LLM) best RPD: {llm_best:+.2f}%   (g1)")
    print(f"gap (LLM prior)        : {llm_best - random_best:+.2f}pp  "
          f"(negative = LLM better)")
    print("=" * 50)


if __name__ == "__main__":
    main()
