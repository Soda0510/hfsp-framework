"""
Stratified train/test split of the Fernández-Viagas 480 benchmark.

Stratum = (n_jobs, n_stages).  Each stratum has 10 instances (idx 1..10):
    - train: idx 1..7   (7 per stratum -> 336 total)
    - test:  idx 8..10  (3 per stratum -> 144 total)

The evolution's actual training subset is a smaller, size-covered sample of the
train set (see ``evolution_train_subset``) chosen so fitness evaluation during
evolution stays fast, while still spanning the size range.

Rationale (see docs/llm_metaheuristic_design.md §7.1):
  - train/test come from the SAME benchmark -> distribution matches,
  - the published UpperBounds are used as the fitness reference (an absolute
    target: seeds sit at +5..10%, giving the evolution a real signal),
  - the test portion is never seen during evolution or fine-tuning.
"""

import json
import re
from typing import Dict, List, Tuple

from .representation import AlgorithmSpec  # noqa: F401  (re-export convenience)

_NAME_RE = re.compile(r"instancia_(\d+)_(\d+)_(\d+)")


def parse_key(name: str) -> Tuple[int, int, int]:
    m = _NAME_RE.search(name)
    if not m:
        raise ValueError(f"Not a valid instancia name: {name!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def stratified_split(all_names: List[str]) -> Tuple[List[str], List[str]]:
    """Split all instance names into (train, test) stratified by (jobs, stages).

    Deterministic: within each stratum the first 7 idx are train, last 3 test.
    """
    strata: Dict[Tuple[int, int], List[str]] = {}
    for name in sorted(all_names):
        n, s, idx = parse_key(name)
        strata.setdefault((n, s), []).append(name)

    train, test = [], []
    for key in sorted(strata):
        names = sorted(strata[key], key=lambda nm: parse_key(nm)[2])
        assert len(names) == 10, f"stratum {key} has {len(names)} (expected 10)"
        train.extend(names[:7])
        test.extend(names[7:])
    return sorted(train), sorted(test)


ALL_SIZES: Tuple[int, ...] = (10, 15, 20, 25, 30, 35, 40, 80, 120, 160, 200, 240)
ALL_STAGES: Tuple[int, ...] = (5, 10, 15, 20)


def evolution_train_subset(
    train_names: List[str],
    sizes: Tuple[int, ...] = ALL_SIZES,
    stages: Tuple[int, ...] = ALL_STAGES,
    per_stratum: int = 1,
) -> List[str]:
    """Stratified subset of the train set spanning the FULL size/stage range.

    Default: one instance per (size, stage) across all 12 sizes (10..240) and
    all 4 stages -> 48 instances, so the training distribution matches the test
    distribution.  Evolution is slower (big instances) but the full run accepts
    it; use smaller ``sizes`` for fast quick iterations.
    """
    by_key: Dict[Tuple[int, int], List[str]] = {}
    for name in train_names:
        n, s, idx = parse_key(name)
        by_key.setdefault((n, s), []).append(name)

    subset = []
    for n in sizes:
        for s in stages:
            key = (n, s)
            if key not in by_key:
                continue
            names = sorted(by_key[key], key=lambda nm: parse_key(nm)[2])
            subset.extend(names[:per_stratum])
    return sorted(subset)


def kfold_split(all_names: List[str], k: int = 5) -> List[Tuple[List[str], List[str]]]:
    """K-fold stratified CV: every instance is used as a test exactly once.

    Returns a list of ``(train, test)`` folds, stratified by (jobs, stages).
    Using all 480 across folds answers "we tested on the whole benchmark".
    """
    strata: Dict[Tuple[int, int], List[str]] = {}
    for name in sorted(all_names):
        n, s, idx = parse_key(name)
        strata.setdefault((n, s), []).append(name)

    folds = [[] for _ in range(k)]
    for key in sorted(strata):
        names = sorted(strata[key], key=lambda nm: parse_key(nm)[2])
        for i, nm in enumerate(names):
            folds[i % k].append(nm)     # round-robin -> each fold has 2 of 10 per stratum

    result = []
    for i in range(k):
        test = folds[i]
        train = [nm for j in range(k) if j != i for nm in folds[j]]
        result.append((sorted(train), sorted(test)))
    return result


def save_split(out_dir: str, all_names: List[str],
               subset_sizes: Tuple[int, ...] = ALL_SIZES,
               subset_stages: Tuple[int, ...] = ALL_STAGES,
               per_stratum: int = 1) -> Dict:
    """Compute the split + evolution subset and save them as text/JSON files."""
    import os
    from pathlib import Path

    train, test = stratified_split(all_names)
    evo_sub = evolution_train_subset(train, subset_sizes, subset_stages, per_stratum)

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.txt").write_text("\n".join(train) + "\n")
    (d / "test.txt").write_text("\n".join(test) + "\n")
    (d / "evolution_train_subset.txt").write_text("\n".join(evo_sub) + "\n")

    summary = {
        "total": len(all_names),
        "train": len(train),
        "test": len(test),
        "evolution_train_subset": len(evo_sub),
        "split_rule": "per (n_jobs, n_stages) stratum: idx 1-7 train, idx 8-10 test",
        "evolution_subset_rule": (
            f"per_stratum={per_stratum}, sizes={list(subset_sizes)}, stages={list(subset_stages)}"),
    }
    (d / "split.json").write_text(json.dumps(summary, indent=2))
    return summary
