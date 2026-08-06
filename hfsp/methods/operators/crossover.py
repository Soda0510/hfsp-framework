"""
Crossover operators for permutation-based representations.

Supported operators:
- OX (Order Crossover)
- PMX (Partially Mapped Crossover)
- PBX (Position-Based Crossover)
- TPX (Two-Point Crossover)
"""

from typing import List
import numpy as np

from .base import Operator


class OrderCrossover(Operator):
    """
    OX: Order Crossover.

    1. Select a subsequence from parent1.
    2. Copy that subsequence to the same positions in the child.
    3. Fill remaining positions with the order of jobs from parent2.
    """

    name = "OX"
    category = "crossover"

    def apply(self, permutation: List[int],
              rng: np.random.Generator) -> List[int]:
        # OX requires two parents. Single-parent call returns a copy.
        return list(permutation)

    def crossover(self, parent1: List[int], parent2: List[int],
                  rng: np.random.Generator) -> tuple:
        """Apply OX to two parents, return two children."""
        n = len(parent1)

        def ox_one(p1, p2):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i > j:
                i, j = j, i

            child = [-1] * n
            # Copy segment from p1
            child[i:j + 1] = p1[i:j + 1]
            # Fill from p2 in order, skipping already-placed jobs
            p2_pos = 0
            for k in range(n):
                if child[k] == -1:
                    while p2[p2_pos % n] in child:
                        p2_pos += 1
                    child[k] = p2[p2_pos % n]
                    p2_pos += 1
            return child

        return ox_one(parent1, parent2), ox_one(parent2, parent1)


class PMXCrossover(Operator):
    """
    PMX: Partially Mapped Crossover.

    1. Select two cut points.
    2. Swap the segments between parents, creating a mapping.
    3. Resolve conflicts using the mapping.
    """

    name = "PMX"
    category = "crossover"

    def apply(self, permutation: List[int],
              rng: np.random.Generator) -> List[int]:
        return list(permutation)

    def crossover(self, parent1: List[int], parent2: List[int],
                  rng: np.random.Generator) -> tuple:
        n = len(parent1)

        def pmx_one(p1, p2):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i > j:
                i, j = j, i

            # Standard PMX: start with a copy of p1,
            # then for each position in [i, j], swap child[k]
            # with the position where p2[k] currently sits.
            child = list(p1)
            for k in range(i, j + 1):
                if child[k] == p2[k]:
                    continue
                pos = child.index(p2[k])
                child[k], child[pos] = child[pos], child[k]
            return child

        return pmx_one(parent1, parent2), pmx_one(parent2, parent1)


class TwoPointCrossover(Operator):
    """
    TPX: Two-Point Crossover.

    Copy the segments outside two cut points from parent1,
    fill the middle segment from parent2 in order.
    """

    name = "TPX"
    category = "crossover"

    def apply(self, permutation: List[int],
              rng: np.random.Generator) -> List[int]:
        return list(permutation)

    def crossover(self, parent1: List[int], parent2: List[int],
                  rng: np.random.Generator) -> tuple:
        n = len(parent1)

        def tpx_one(p1, p2):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i > j:
                i, j = j, i

            child = [-1] * n
            # Copy ends from p1
            child[:i] = p1[:i]
            child[j + 1:] = p1[j + 1:]

            # Fill middle from p2 order
            p2_idx = 0
            for k in range(i, j + 1):
                while p2[p2_idx] in child:
                    p2_idx += 1
                child[k] = p2[p2_idx]
                p2_idx += 1

            return child

        return tpx_one(parent1, parent2), tpx_one(parent2, parent1)
