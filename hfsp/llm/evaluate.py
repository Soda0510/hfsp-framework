"""
Fitness evaluation for AlgorithmSpecs.

The fitness of a candidate algorithm is its average makespan (or RPD) over a
fixed set of training instances, evaluated with FIXED seeds so that results
are reproducible and comparisons between candidate algorithms are fair.
"""

from typing import Callable, List, Optional
import numpy as np

from ..core.instance import HFSPInstance
from .executor import ConfigurableMetaheuristic
from .representation import AlgorithmSpec


def make_fitness(
    instances: List[HFSPInstance],
    n_runs: int = 1,
    seed_base: int = 0,
    references: Optional[List[float]] = None,
    time_limit: float = float("inf"),
) -> Callable[[AlgorithmSpec], float]:
    """
    Build a fitness function over a spec (lower is better).

    If ``references`` (one best-known makespan per instance) is given, the
    returned value is the mean RPD (%): ``(mean_i - ref_i) / ref_i * 100``
    averaged over instances.  Otherwise it is the raw mean makespan.

    Seeds are fixed per (instance, run) so every spec is evaluated under the
    same random draws — differences come only from algorithm structure.
    """
    if references is not None and len(references) != len(instances):
        raise ValueError("references must have one value per instance")

    def fitness(spec: AlgorithmSpec) -> float:
        inst_means = []
        for i, inst in enumerate(instances):
            makespans = []
            for run in range(n_runs):
                rng = np.random.default_rng(seed_base + i * 1000 + run)
                sol = ConfigurableMetaheuristic(
                    spec, rng=rng, time_limit=time_limit
                ).solve(inst)
                makespans.append(sol.makespan)
            inst_means.append(float(np.mean(makespans)))

        if references is None:
            return float(np.mean(inst_means))

        rpd = [
            (inst_means[i] - references[i]) / max(references[i], 1e-9) * 100.0
            for i in range(len(instances))
        ]
        return float(np.mean(rpd))

    return fitness


def per_instance_best(instances: List[HFSPInstance],
                      specs: List[AlgorithmSpec],
                      n_runs: int = 1,
                      seed_base: int = 0,
                      time_limit: float = float("inf")) -> List[float]:
    """
    Compute one reference (best makespan) per instance across a set of specs.
    Used to turn raw makespans into RPD-based fitness.
    """
    refs = []
    for i, inst in enumerate(instances):
        best = float("inf")
        for spec in specs:
            for run in range(n_runs):
                rng = np.random.default_rng(seed_base + i * 1000 + run)
                sol = ConfigurableMetaheuristic(spec, rng=rng,
                                               time_limit=time_limit).solve(inst)
                best = min(best, sol.makespan)
        refs.append(best)
    return refs
