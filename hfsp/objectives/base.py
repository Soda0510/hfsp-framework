"""
Abstract base class for objective functions.
"""

from abc import ABC, abstractmethod
from ..core.solution import ScheduleSolution


class ObjectiveFunction(ABC):
    """Base class for all objective functions.

    Each objective function evaluates a ScheduleSolution and returns
    a scalar value (to be minimized).
    """

    name: str = "base"

    @abstractmethod
    def compute(self, solution: ScheduleSolution) -> float:
        """Evaluate the solution and return the objective value."""
        ...
