"""
Convergence tracker for recording per-iteration best fitness.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class ConvergenceTracker:
    """
    Records the best objective value at each iteration/generation.

    Usage:
        tracker = ConvergenceTracker()
        ...
        tracker.record(best_fitness, generation)
        ...
        df = tracker.to_dataframe()
    """

    def __init__(self):
        self.best_fitness: List[float] = []
        self.generations: List[int] = []

    def record(self, fitness: float, generation: int = -1):
        """Record the best fitness at a given generation."""
        self.best_fitness.append(fitness)
        if generation < 0:
            generation = len(self.generations)
        self.generations.append(generation)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "generation": self.generations,
            "best_fitness": self.best_fitness,
        })

    def to_list(self) -> List[float]:
        return list(self.best_fitness)
