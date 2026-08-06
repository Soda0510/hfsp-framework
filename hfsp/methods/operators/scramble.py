"""
Scramble mutation: randomly shuffles a subsequence.
"""

from typing import List
import numpy as np

from .base import Operator


class ScrambleOperator(Operator):
    """Scramble (randomly shuffle) a subsequence of the permutation."""

    name = "scramble"
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

        subseq = result[i:j + 1]
        rng.shuffle(subseq)
        result[i:j + 1] = subseq
        return result
