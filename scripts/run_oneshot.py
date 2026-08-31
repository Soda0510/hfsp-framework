"""
Ablation F: one-shot LLM design vs the evolved algorithm.

Asks the base LLM ONCE (zero-shot, no evolution, no feedback) to design an
HFSP metaheuristic, then evaluates it on a quick test subset with the same
protocol as the evolved g1.  This isolates the contribution of the evolutionary
search over the LLM's raw pretrained prior.

Usage:
    .venv/bin/python scripts/run_oneshot.py [--rho 1] [--sizes 10,20,40,80,160]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from hfsp.io import SevilleReader, SevilleReference
from hfsp.llm import (
    AlgorithmSpec, ConfigurableMetaheuristic, OllamaClient,
    spec_from_dict, SYSTEM_PROMPT, build_design_prompt,
)
from hfsp.methods.metaheuristics import IteratedGreedy

SEVILLE = "benchmarks/seville"
BEST_SPEC = "results/llm/base_run/best_spec.json"
TEST_LIST = "benchmarks/seville/split/test.txt"
OUT_DIR = Path("results/llm/oneshot")


def load_test_sample(sizes, per_size=2):
    reader = SevilleReader(SEVILLE)
    test_names = [ln.strip() for ln in open(TEST_LIST) if ln.strip()]
    from hfsp.llm.dataset import parse_key
    chosen = []
    for n in sorted(sizes):
        chosen.extend([nm for nm in test_names if parse_key(nm)[0] == n][:per_size])
    return chosen


def design_oneshot(client, max_retries=3):
    """Ask the base LLM for a fresh AlgorithmSpec."""
    for attempt in range(max_retries):
        try:
            raw = client.chat_json(SYSTEM_PROMPT, build_design_prompt())
            return spec_from_dict(raw, name="oneshot")
        except Exception as e:
            print(f"  [oneshot] attempt {attempt + 1} failed: {str(e)[:80]}")
    raise RuntimeError("could not obtain a valid one-shot spec")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--sizes", default="10,20,40,80,160")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client = OllamaClient()
    print("asking base LLM for a one-shot HFSP design...")
    oneshot = design_oneshot(client)
    print(f"one-shot spec: {oneshot.name} | destroy={oneshot.use_destroy_repair}, "
          f"init={oneshot.initializer}, accept={oneshot.acceptance}, "
          f"ls={oneshot.use_local_search}, pop={oneshot.population_mode}")
    (OUT_DIR / "oneshot_spec.json").write_text(json.dumps(oneshot.to_dict(), indent=2))

    g1 = AlgorithmSpec(**json.load(open(BEST_SPEC)))
    names = load_test_sample([int(s) for s in args.sizes.split(",")])
    reader = SevilleReader(SEVILLE)
    ref = SevilleReference(f"{SEVILLE}/references/UpperBounds_01_April_2019.xlsx")

    algos = {"oneshot": oneshot, "g1": g1}
    results = {k: [] for k in algos}
    ig_scores = []
    head = {"g1": 0, "ig": 0}   # oneshot wins vs each

    t0 = time.perf_counter()
    for name in names:
        inst = reader.load(name)
        bk = ref.best_known(name)
        tl = args.rho * inst.num_jobs * inst.total_machines / 1000.0
        scores = {}
        for k, spec in algos.items():
            rng = np.random.default_rng(42)
            scores[k] = ConfigurableMetaheuristic(spec, rng=rng, time_limit=tl).solve(inst).makespan
        rng = np.random.default_rng(42)
        scores["ig"] = IteratedGreedy(max_iterations=50000, use_local_search=True,
                                      rng=rng, time_limit=tl).solve(inst).makespan
        ig_scores.append((scores["ig"] - bk) / bk * 100.0)
        for k in algos:
            results[k].append((scores[k] - bk) / bk * 100.0)
        for k in ("g1", "ig"):
            if scores["oneshot"] < scores[k] - 1e-9:
                head[k] += 1
            elif scores["oneshot"] > scores[k] + 1e-9:
                pass  # oneshot loses, not counted in wins

    print(f"\ntest sample: {len(names)} instances | rho={args.rho} | took {time.perf_counter()-t0:.0f}s")
    print(f"\n{'algorithm':<10}{'mean RPD':>10}{'min':>9}{'max':>9}")
    print("-" * 38)
    for k in algos:
        v = results[k]
        print(f"{k:<10}{np.mean(v):>10.2f}{np.min(v):>9.2f}{np.max(v):>9.2f}")
    print(f"IG            {np.mean(ig_scores) if ig_scores else 'n/a':>10}")

    print(f"\noneshot head-to-head wins (of {len(names)}): vs g1 = {head['g1']}, vs IG = {head['ig']}")
    (OUT_DIR / "oneshot_eval.json").write_text(json.dumps(
        {"names": names,
         "oneshot": {k: float(np.mean(v)) for k, v in results.items()},
         "wins": head}, indent=2))


if __name__ == "__main__":
    main()
