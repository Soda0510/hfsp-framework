"""
Configuration dataclasses for experiments and algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AlgorithmConfig:
    """Configuration for a single algorithm run."""

    name: str  # "NEH", "GA", "SA", "IG", "MILP"

    # GA parameters
    population_size: int = 80
    crossover_prob: float = 0.8
    mutation_prob: float = 0.3
    max_generations: int = 500
    elite_size: int = 2

    # SA parameters
    initial_temperature: float = 100.0
    cooling_rate: float = 0.97
    max_total_iterations: int = 50000

    # IG parameters
    ig_iterations: int = 2000
    ig_use_local_search: bool = True

    # General
    time_limit: float = float("inf")


@dataclass
class ExperimentConfig:
    """Configuration for a batch experiment."""

    name: str = "experiment"
    description: str = ""
    instances: List[str] = field(default_factory=list)  # instance names or patterns
    algorithms: List[AlgorithmConfig] = field(default_factory=list)
    num_runs: int = 10
    seed_base: int = 20240101
    output_dir: str = "results"
    save_convergence: bool = True
