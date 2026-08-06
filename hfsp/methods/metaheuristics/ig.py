"""
Iterated Greedy (IG) algorithm for HFSP.

Based on the framework by Ruiz & Stützle (2007), adapted for HFSP:
1. Generate initial solution (NEH).
2. Destruction: remove d random jobs from the permutation.
3. Construction: reinsert removed jobs at best positions (NEH-style).
4. Local search (optional).
5. Acceptance: keep new solution if better, or with temperature-based probability.
"""

from typing import Optional
import numpy as np
import time

from ..base import Method
from ...core.instance import HFSPInstance
from ...core.solution import ScheduleSolution
from ...core.decoder import ListSchedulingDecoder
from ..heuristics import neh_heuristic
from ..operators import local_search


class IteratedGreedy(Method):
    """
    Iterated Greedy for HFSP.

    Parameters
    ----------
    destruction_size : int
        Number of jobs to remove (d). Default: n_jobs // 10 (min 2).
    temperature : float
        Acceptance temperature; computed adaptively if None.
    max_iterations : int
        Maximum iterations.
    use_local_search : bool
        Apply local search after reconstruction.
    decoder : ListSchedulingDecoder, optional
    rng : np.random.Generator, optional
    time_limit : float
    """

    name = "IG"

    def __init__(
        self,
        destruction_size: int = None,
        temperature: float = None,
        max_iterations: int = 2000,
        use_local_search: bool = True,
        decoder: ListSchedulingDecoder = None,
        rng: np.random.Generator = None,
        time_limit: float = float("inf"),
    ):
        super().__init__(rng=rng, time_limit=time_limit)
        self.destruction_size = destruction_size
        self.temperature_user = temperature
        self.max_iterations = max_iterations
        self.use_local_search = use_local_search
        self.decoder = decoder if decoder is not None else ListSchedulingDecoder()

    def solve(self, instance: HFSPInstance) -> ScheduleSolution:
        self.start_time = time.perf_counter()
        self.convergence = []
        n_jobs = instance.num_jobs

        # Destruction size: 10% of jobs, min 2
        d = self.destruction_size
        if d is None:
            d = max(2, n_jobs // 10)

        # Initial solution (NEH)
        current = neh_heuristic(instance, self.decoder, self.rng)
        best = current.copy()
        self.convergence.append(best.makespan)

        # Adaptive temperature
        if self.temperature_user is not None:
            T = self.temperature_user
        else:
            # T such that a solution 5% worse has ~50% acceptance probability
            T = 0.05 * best.makespan / np.log(2) if best.makespan > 0 else 1.0

        for iteration in range(self.max_iterations):
            if self._check_time():
                break

            # Destruction: remove d random jobs
            perm = list(current.permutation)
            removed = []
            for _ in range(d):
                if len(perm) <= 1:
                    break
                idx = int(self.rng.integers(0, len(perm)))
                removed.append(perm.pop(idx))

            # Construction: NEH-style reinsertion
            for job in removed:
                best_pos = 0
                best_ms = float("inf")
                for pos in range(len(perm) + 1):
                    candidate = perm[:pos] + [job] + perm[pos:]
                    sol = self.decoder.decode(instance, candidate, self.rng)
                    if sol.makespan < best_ms:
                        best_ms = sol.makespan
                        best_pos = pos
                perm = perm[:best_pos] + [job] + perm[best_pos:]

            new_sol = self.decoder.decode(instance, perm, self.rng)

            # Local search
            if self.use_local_search:
                new_sol = local_search(
                    new_sol, self.decoder,
                    max_iterations=50, strategy="first_improvement",
                    rng=self.rng,
                )

            # Acceptance
            delta = new_sol.makespan - current.makespan
            if delta < 0 or self.rng.random() < np.exp(-delta / max(T, 1e-12)):
                current = new_sol
                if current.makespan < best.makespan - 1e-12:
                    best = current.copy()

            self.convergence.append(best.makespan)

        best.method = self.name
        return best
