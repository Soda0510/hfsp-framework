"""
Insert mutation operator: removes a job and reinserts it at another position.
"""

from typing import List
import numpy as np

from .base import Operator


class InsertOperator(Operator):
    """Remove a job at one position and insert it at another."""

    name = "insert"
    category = "mutation"

    def apply(self, permutation: List[int],
              rng: np.random.Generator) -> List[int]:
        n = len(permutation)
        if n < 2:
            return list(permutation)

        result = list(permutation)
        from_pos = int(rng.integers(0, n))
        job = result.pop(from_pos)
        to_pos = int(rng.integers(0, n - 1))
        result.insert(to_pos, job)
        return result
