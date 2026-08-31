"""
Block move mutation operator: pick a contiguous block of jobs and relocate it.

A "block" move is a classic neighborhood for permutation scheduling: moving a
contiguous subsequence as a unit often perturbs the schedule more meaningfully
than a single swap/insert (helps escape local optima).
"""

from typing import List
import numpy as np

from .base import Operator


class BlockOperator(Operator):
    """Move a random contiguous block to a new position."""

    name = "block"
    category = "mutation"

    def apply(self, permutation: List[int],
              rng: np.random.Generator) -> List[int]:
        n = len(permutation)
        if n < 3:
            return list(permutation)

        # Random block length in [1, max(2, n//3)] and start position.
        max_len = max(2, n // 3)
        b_len = int(rng.integers(1, max_len + 1))
        if b_len >= n:
            return list(permutation)
        start = int(rng.integers(0, n - b_len + 1))
        block = permutation[start:start + b_len]

        # Remaining sequence; insert the block at a new (different) position.
        rest = permutation[:start] + permutation[start + b_len:]
        if len(rest) == 0:
            return list(permutation)
        new_pos = int(rng.integers(0, len(rest) + 1))
        # Avoid re-inserting at the exact same place.
        guard = 0
        while (new_pos == start and guard < 5):
            new_pos = int(rng.integers(0, len(rest) + 1))
            guard += 1
        result = rest[:new_pos] + block + rest[new_pos:]
        return result
