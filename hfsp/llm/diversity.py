"""
Diversity metrics for the algorithm-design population (阶段4, anti-collapse).

The collapse failure mode is "one algorithm is always best, population
homogenizes".  A similarity penalty on selection keeps structurally-different
candidates alive even if slightly worse on fitness.
"""

from typing import List
from .representation import AlgorithmSpec

_NUMERIC_FIELDS = ["max_iterations", "population_size", "local_search_iterations",
                   "crossover_prob", "mutation_prob", "cooling_rate"]


def component_similarity(a: AlgorithmSpec, b: AlgorithmSpec) -> float:
    """Structural similarity in [0,1] between two algorithm designs.

    0.7 * Jaccard over discrete components + 0.3 * avg normalized numeric
    closeness.  Higher = more similar.
    """
    disc_a = {
        a.initializer, a.acceptance,
        a.population_mode, a.use_destroy_repair, a.use_local_search,
        frozenset(a.mutation_operators), frozenset(a.crossover_operators),
    }
    disc_b = {
        b.initializer, b.acceptance,
        b.population_mode, b.use_destroy_repair, b.use_local_search,
        frozenset(b.mutation_operators), frozenset(b.crossover_operators),
    }
    union = disc_a | disc_b
    jaccard = len(disc_a & disc_b) / len(union) if union else 1.0

    # normalized numeric closeness (avg of 1 - |diff|/range-ish)
    num_scores = []
    for f in _NUMERIC_FIELDS:
        va, vb = getattr(a, f), getattr(b, f)
        if va is None or vb is None:
            num_scores.append(1.0 if va == vb else 0.0)
            continue
        denom = max(abs(va), abs(vb), 1e-9)
        num_scores.append(1.0 - min(1.0, abs(va - vb) / denom))
    num_sim = sum(num_scores) / len(num_scores) if num_scores else 1.0

    return 0.7 * jaccard + 0.3 * num_sim


def max_similarity(spec: AlgorithmSpec, population: List[AlgorithmSpec]) -> float:
    """Highest similarity of ``spec`` to any member of ``population``."""
    if not population:
        return 0.0
    return max(component_similarity(spec, other) for other in population)
