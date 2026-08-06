"""
Abstract base class for all solution methods (heuristics + metaheuristics).
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
import time

from ..core.instance import HFSPInstance
from ..core.solution import ScheduleSolution


class Method(ABC):
    """
    Base class for all solution methods.

    All methods must implement `solve(instance) -> ScheduleSolution`.
    """

    name: str = "base"

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        time_limit: float = float("inf"),
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.time_limit = time_limit
        self.start_time: float = 0.0
        self.best_solution: Optional[ScheduleSolution] = None
        self.convergence: List[float] = []  # best objective per iteration

    @abstractmethod
    def solve(self, instance: HFSPInstance) -> ScheduleSolution:
        """Solve the instance and return the best schedule found."""
        ...

    def _check_time(self) -> bool:
        """Return True if time limit has been exceeded."""
        if self.time_limit == float("inf"):
            return False
        return time.perf_counter() - self.start_time >= self.time_limit

    def _record_convergence(self, obj: float):
        """Record the best objective value at this iteration."""
        self.convergence.append(obj)
