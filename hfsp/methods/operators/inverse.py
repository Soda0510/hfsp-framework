"""
Inverse mutation operator: reverses a random subsequence.
"""

from typing import List
import numpy as np

from .base import Operator


class InverseOperator(Operator):
    """Reverse a random subsequence of the permutation."""

    name = "inverse"
    category = "mutation"

    def apply(self, permutation: List[int],
              rng: np.random.Generator) -> List[int]:
        n = len(permutation)
        if n < 2:
            return list(permutation)

        result = list(permutation)
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i > j:
            i, j = j, i
        if i == j:
            if i > 0:
                i -= 1
            elif j < n - 1:
                j += 1

        result[i:j + 1] = reversed(result[i:j + 1])
        return result
