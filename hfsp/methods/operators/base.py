"""
Abstract base class for neighborhood operators and operator registry.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


# Global operator registry
OPERATOR_REGISTRY: dict[str, "Operator"] = {}


class Operator(ABC):
    """
    Base class for all neighborhood operators on job permutations.

    Attributes
    ----------
    name : str
        Unique operator identifier (e.g., "swap", "N1_Cswap").
    category : str
        Either "mutation", "crossover", or "local_search".
    """

    name: str = "base"
    category: str = "mutation"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name != "base":
            OPERATOR_REGISTRY[cls.name] = cls

    @abstractmethod
    def apply(self, permutation: List[int],
              rng: np.random.Generator) -> List[int]:
        """
        Apply the operator and return a NEW permutation.
        Does NOT mutate the input.

        Returns
        -------
        list[int]
            New permutation.
        """
        ...


def get_operator(name: str) -> Operator:
    """Look up an operator by name in the registry."""
    if name not in OPERATOR_REGISTRY:
        raise KeyError(
            f"Operator '{name}' not found. Available: {list(OPERATOR_REGISTRY.keys())}"
        )
    return OPERATOR_REGISTRY[name]()


def list_operators(category: str = None) -> List[str]:
    """List all registered operators, optionally filtered by category."""
    if category:
        return [n for n, cls in OPERATOR_REGISTRY.items()
                if cls.category == category]
    return list(OPERATOR_REGISTRY.keys())
