"""
Seed AlgorithmSpecs: known-good algorithms that initialize the population.

Each seed spec must execute through ConfigurableMetaheuristic and, ideally,
reproduce its hand-crafted counterpart (verified in scripts/verify_executor.py).

Note: DPSO is NOT representable in the current design space (velocity-based,
outside the four search templates), so it stays an external comparison
baseline rather than a seed spec.
"""

from .representation import AlgorithmSpec


def ga_seed_spec() -> AlgorithmSpec:
    """Mirrors GeneticAlgorithm(defaults) exactly (random init)."""
    return AlgorithmSpec(
        name="GA",
        population_mode=True,
        use_destroy_repair=False,
        initializer="random",            # GA uses fully random init
        mutation_operators=["swap", "insert", "inverse"],
        crossover_operators=["ox", "pmx", "two_point"],
        crossover_prob=0.8,
        mutation_prob=0.3,
        acceptance="only_better",
        use_local_search=False,
        population_size=80,
        max_iterations=500,              # = generations
        elite_size=2,
        tournament_size=2,
    )


def ig_seed_spec() -> AlgorithmSpec:
    """Mirrors IteratedGreedy(defaults) exactly."""
    return AlgorithmSpec(
        name="IG",
        population_mode=False,
        use_destroy_repair=True,
        initializer="neh",
        mutation_operators=["swap", "insert", "inverse"],
        crossover_operators=["ox"],      # unused in this template
        acceptance="metropolis",         # IG: delta<0 or exp(-delta/T)
        temperature=None,                # adaptive, like IG
        use_local_search=True,
        local_search_iterations=50,      # IG uses max_iterations=50
        max_iterations=2000,
        destruction_size=None,           # auto: max(2, n//10)
    )


def sa_seed_spec() -> AlgorithmSpec:
    """SA-like single-solution search (comparable, not identical, to SA)."""
    return AlgorithmSpec(
        name="SA-like",
        population_mode=False,
        use_destroy_repair=False,
        initializer="neh",
        mutation_operators=["swap", "insert", "inverse"],
        crossover_operators=["ox"],
        acceptance="metropolis",
        temperature=None,                # adaptive
        cooling_rate=0.99,
        use_local_search=False,
        max_iterations=10000,
    )


def neh_ls_seed_spec() -> AlgorithmSpec:
    """Pure constructive: NEH + local search (max_iterations=0 path)."""
    return AlgorithmSpec(
        name="NEH+LS",
        population_mode=False,
        use_destroy_repair=False,
        initializer="neh",
        mutation_operators=["swap", "insert", "inverse"],
        crossover_operators=["ox"],
        acceptance="only_better",
        use_local_search=True,
        local_search_iterations=100,
        max_iterations=0,                # -> constructive path
    )


def default_seed_specs():
    """The canonical seed set used to initialize the population."""
    return [ga_seed_spec(), ig_seed_spec(), sa_seed_spec(), neh_ls_seed_spec()]
