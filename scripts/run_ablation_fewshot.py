"""
Ablation B: few-shot in-context demonstrations (no fine-tuning).

Shows the LLM 2-3 real, successful (parent -> improved child) mutation examples
from the evolution log, then runs a short evolution.  If few-shot helps, it is
cheap evidence that the "learning signal" matters — a probe for M8 fine-tuning.

Usage:
    .venv/bin/python scripts/run_ablation_fewshot.py [--gen 6] [--pop 8] [--rho 0.3]
"""

import argparse
import csv
import json
import time
from pathlib import Path

from hfsp.io import SevilleReader, SevilleReference
from hfsp.llm import (
    make_fitness, EvolutionarySearch, OllamaClient, LLMMutator, ReflectionEngine,
)
from hfsp.llm.seeds import default_seed_specs

SEVILLE = "benchmarks/seville"
REF_XLSX = "benchmarks/seville/references/UpperBounds_01_April_2019.xlsx"
SPLIT_SUBSET = "benchmarks/seville/split/evolution_train_subset.txt"
MUT_LOG = "results/llm/base_run/mutation_log.jsonl"
TOP10_CSV = "results/llm/e_top10/top10.csv"
OUT_DIR = Path("results/llm/fewshot")


def curate_examples():
    """Pick 3 structurally-diverse successful (parent -> child) pairs."""
    best_rpd = {}
    with open(TOP10_CSV) as f:
        for r in csv.DictReader(f):
            nm, rpd = r["name"], float(r["rpd"])
            if nm not in best_rpd or rpd < best_rpd[nm]:
                best_rpd[nm] = rpd

    rows = [json.loads(l) for l in open(MUT_LOG)]
    curated = []
    seen = set()
    for r in rows:
        cn = r["child"].get("name")
        c_sig = json.dumps(r["child"], sort_keys=True)
        if (cn in best_rpd and c_sig not in seen
                and best_rpd[cn] < r["parent_fitness"] - 1e-6):
            curated.append({
                "parent": r["parent"],
                "child": r["child"],
                "parent_fitness": r["parent_fitness"],
                "child_fitness": best_rpd[cn],
                "type": (r["child"]["population_mode"], r["child"]["use_destroy_repair"]),
            })
            seen.add(c_sig)

    # pick one per (pop, destroy) combination for structural diversity
    chosen, used_types = [], set()
    for ex in curated:
        if ex["type"] not in used_types:
            chosen.append(ex)
            used_types.add(ex["type"])
        if len(chosen) == 3:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=6)
    parser.add_argument("--pop", type=int, default=8)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = curate_examples()
    print(f"few-shot examples: {len(examples)}")
    for ex in examples:
        print(f"  {ex['parent'].get('name')}({ex['parent_fitness']:+.2f}%) -> "
              f"{ex['child'].get('name')}({ex['child_fitness']:+.2f}%)  "
              f"type(pop={int(ex['child']['population_mode'])},destroy={int(ex['child']['use_destroy_repair'])})")
    (OUT_DIR / "examples.json").write_text(json.dumps(examples, indent=2))

    train_names = [ln.strip() for ln in open(SPLIT_SUBSET) if ln.strip()]
    reader = SevilleReader(SEVILLE)
    ref = SevilleReference(REF_XLSX)
    instances = [reader.load(n) for n in train_names]
    refs = [ref.best_known(n) for n in train_names]
    budget = max(args.rho * i.num_jobs * i.total_machines / 1000.0 for i in instances)
    fitness = make_fitness(instances, n_runs=1, references=refs, seed_base=0,
                           time_limit=budget)

    client = OllamaClient()
    mutator = LLMMutator(client, max_retries=2, verbose=True, examples=examples)
    engine = ReflectionEngine(client, verbose=True)

    t0 = time.perf_counter()
    evo = EvolutionarySearch(
        fitness_fn=fitness, mutator=mutator.mutate,
        population_size=args.pop, elite_size=2, generations=args.gen,
        seed=args.seed, seed_specs=default_seed_specs(),
        use_reflection=True, reflection_engine=engine, log_mutations=True,
    )
    result = evo.run()
    elapsed = time.perf_counter() - t0

    (OUT_DIR / "best_spec.json").write_text(
        json.dumps(result["best_spec"].to_dict(), indent=2))
    (OUT_DIR / "convergence.json").write_text(
        json.dumps({"history": result["history"]}, indent=2))

    base_conv = json.load(open("results/llm/base_run/convergence.json"))
    print(f"\n{'='*50}")
    print(f"few-shot best RPD: {result['best_fitness']:+.2f}%   "
          f"(gen={args.gen}, took {elapsed:.0f}s)")
    print(f"zero-shot best RPD (g1): {base_conv['history'][-1]:+.2f}%")
    print(f"gap: {result['best_fitness'] - base_conv['history'][-1]:+.2f}pp "
          f"(negative = few-shot better)")
    print(f"convergence: {[f'{h:+.2f}' for h in result['history']]}")
    print(f"LLM stats: {mutator.stats}")


if __name__ == "__main__":
    main()
