"""
Swap mutation operator: swaps two randomly selected positions.
"""

from typing import List
import numpy as np

from .base import Operator


class SwapOperator(Operator):
    """Swap two randomly selected jobs in the permutation."""

    name = "swap"
    category = "mutation"

    def apply(self, permutation: List[int],
              rng: np.random.Generator) -> List[int]:
        n = len(permutation)
        if n < 2:
            return list(permutation)

        result = list(permutation)
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        # Ensure i != j
        while i == j:
            j = int(rng.integers(0, n))
        result[i], result[j] = result[j], result[i]
        return result
