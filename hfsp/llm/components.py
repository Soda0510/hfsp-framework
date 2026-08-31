"""
Component pool: maps AlgorithmSpec field values to concrete implementations.

The design space is intentionally bounded to the operators / heuristics already
available in ``hfsp`` so that any valid spec is executable and reproducible.
New components can be added here (or by registering more operators) to widen
the design space.
"""

from typing import List, Optional, Callable

from ..methods.heuristics import (
    neh_heuristic, spt_heuristic, lpt_heuristic, palmer_heuristic, cds_heuristic,
)
from ..methods.operators import (
    SwapOperator,
    InsertOperator,
    InverseOperator,
    ScrambleOperator,
    BlockOperator,
    OrderCrossover,
    PMXCrossover,
    TwoPointCrossover,
)
from .representation import AlgorithmSpec


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Constructive initializers.  "random" has no entry: the executor handles it
# directly (a random permutation) and is therefore not listed here.
INITIALIZERS: dict[str, Callable] = {
    "neh": neh_heuristic,
    "spt": spt_heuristic,
    "lpt": lpt_heuristic,
    "palmer": palmer_heuristic,
    "cds": cds_heuristic,
}

MUTATION_OPERATORS: dict[str, type] = {
    "swap": SwapOperator,
    "insert": InsertOperator,
    "inverse": InverseOperator,
    "scramble": ScrambleOperator,
    "block": BlockOperator,
}

CROSSOVER_OPERATORS: dict[str, type] = {
    "ox": OrderCrossover,
    "pmx": PMXCrossover,
    "two_point": TwoPointCrossover,
}

ACCEPTANCE_CRITERIA: List[str] = [
    "always", "only_better", "metropolis", "threshold", "record_to_record",
]


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def resolve_initializer(name: str) -> Optional[Callable]:
    """Return the initializer callable, or None for the built-in "random"."""
    return INITIALIZERS.get(name)


def resolve_mutation(name: str):
    """Return a mutation Operator instance by name."""
    cls = MUTATION_OPERATORS.get(name)
    if cls is None:
        raise ValueError(f"Unknown mutation operator '{name}'. "
                         f"Available: {sorted(MUTATION_OPERATORS)}")
    return cls()


def resolve_crossover(name: str):
    """Return a crossover Operator instance by name."""
    cls = CROSSOVER_OPERATORS.get(name)
    if cls is None:
        raise ValueError(f"Unknown crossover operator '{name}'. "
                         f"Available: {sorted(CROSSOVER_OPERATORS)}")
    return cls()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def repair_spec(spec: AlgorithmSpec) -> AlgorithmSpec:
    """Patch common cross-field inconsistencies so the spec passes validation.

    Useful for specs produced by an LLM or random sampling, which may set
    fields that are individually legal but inconsistent together.
    """
    if spec.elite_size >= spec.population_size:
        spec.elite_size = max(1, spec.population_size - 1)
    if spec.population_mode and not spec.crossover_operators:
        spec.crossover_operators = ["ox"]
    if not spec.mutation_operators:
        spec.mutation_operators = ["swap"]
    if spec.acceptance == "threshold" and spec.threshold is None:
        spec.threshold = 0.0
    if spec.acceptance != "threshold":
        spec.threshold = None
    if spec.acceptance != "metropolis":
        spec.temperature = None
    # Bound local search: it is the main overrun culprit on large instances
    # (each LS iteration is an expensive decode), so cap its iterations.
    if spec.local_search_iterations > 120:
        spec.local_search_iterations = 120
    return spec


def validate_spec(spec: AlgorithmSpec) -> List[str]:
    """Return a list of validation errors for a spec (empty if valid)."""
    errors: List[str] = []

    if spec.initializer not in ("random", *INITIALIZERS.keys()):
        errors.append(f"initializer '{spec.initializer}' not allowed "
                      f"(allowed: random, {sorted(INITIALIZERS)})")

    for op in spec.mutation_operators:
        if op not in MUTATION_OPERATORS:
            errors.append(f"mutation operator '{op}' not in pool "
                          f"({sorted(MUTATION_OPERATORS)})")

    for op in spec.crossover_operators:
        if op not in CROSSOVER_OPERATORS:
            errors.append(f"crossover operator '{op}' not in pool "
                          f"({sorted(CROSSOVER_OPERATORS)})")

    if spec.acceptance not in ACCEPTANCE_CRITERIA:
        errors.append(f"acceptance '{spec.acceptance}' not allowed "
                      f"({ACCEPTANCE_CRITERIA})")

    if not spec.mutation_operators:
        errors.append("at least one mutation operator is required")

    if spec.population_mode and not spec.crossover_operators:
        errors.append("population_mode requires at least one crossover operator")

    if spec.population_size < 2:
        errors.append("population_size must be >= 2")

    if spec.elite_size >= spec.population_size:
        errors.append("elite_size must be < population_size")

    if spec.max_iterations < 0:
        errors.append("max_iterations must be >= 0")

    if spec.acceptance == "threshold" and spec.threshold is None:
        errors.append("acceptance='threshold' requires a threshold value")

    return errors
