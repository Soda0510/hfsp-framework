"""
Discrete Particle Swarm Optimization (DPSO) for HFSP.

Adapted for permutation-based representation using position-based
ordering and velocity as swap probability.

Each particle:
  - position = permutation (job ordering)
  - pbest = best permutation found by this particle
  - gbest = best permutation in the swarm

Velocity: a weight for each job indicating how strongly to prioritize
moving it toward pbest/gbest positions.
"""

from typing import List, Optional
import numpy as np
import time
import copy

from ..base import Method
from ...core.instance import HFSPInstance
from ...core.solution import ScheduleSolution
from ...core.decoder import ListSchedulingDecoder
from ..heuristics import neh_heuristic


class DiscretePSO(Method):
    """
    Discrete PSO for permutation-based HFSP.

    Parameters
    ----------
    swarm_size : int
        Number of particles.
    max_iterations : int
        Maximum iterations.
    w : float
        Inertia weight (probability of keeping current position).
    c1 : float
        Cognitive acceleration (weight toward pbest).
    c2 : float
        Social acceleration (weight toward gbest).
    decoder : ListSchedulingDecoder, optional
    rng : np.random.Generator, optional
    time_limit : float
    """

    name = "DPSO"

    def __init__(
        self,
        swarm_size: int = 50,
        max_iterations: int = 500,
        w: float = 0.5,
        c1: float = 0.3,
        c2: float = 0.2,
        decoder: ListSchedulingDecoder = None,
        rng: np.random.Generator = None,
        time_limit: float = float("inf"),
    ):
        super().__init__(rng=rng, time_limit=time_limit)
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.decoder = decoder if decoder is not None else ListSchedulingDecoder()

    def solve(self, instance: HFSPInstance) -> ScheduleSolution:
        self.start_time = time.perf_counter()
        self.convergence = []
        n_jobs = instance.num_jobs

        # Initialize swarm with NEH seed + random particles
        neh_sol = neh_heuristic(instance, self.decoder, self.rng)
        particles = [list(neh_sol.permutation)]

        for _ in range(self.swarm_size - 1):
            perm = list(range(n_jobs))
            self.rng.shuffle(perm)
            particles.append(perm)

        # Evaluate initial positions
        fitness = []
        for perm in particles:
            sol = self.decoder.decode(instance, perm, self.rng)
            fitness.append(sol.makespan)

        pbest = [list(p) for p in particles]  # personal best positions
        pbest_fitness = list(fitness)

        # Global best
        gbest_idx = int(np.argmin(pbest_fitness))
        gbest = list(pbest[gbest_idx])
        gbest_fitness = pbest_fitness[gbest_idx]

        best_solution = self.decoder.decode(instance, gbest, self.rng)
        self.convergence.append(gbest_fitness)

        for iteration in range(self.max_iterations):
            if self._check_time():
                break

            for i in range(self.swarm_size):
                new_perm = self._update_particle(
                    particles[i], pbest[i], gbest, n_jobs
                )
                particles[i] = new_perm

                new_sol = self.decoder.decode(instance, new_perm, self.rng)
                new_fitness = new_sol.makespan

                # Update personal best
                if new_fitness < pbest_fitness[i]:
                    pbest[i] = list(new_perm)
                    pbest_fitness[i] = new_fitness

                    # Update global best
                    if new_fitness < gbest_fitness:
                        gbest = list(new_perm)
                        gbest_fitness = new_fitness
                        best_solution = new_sol.copy()

            self.convergence.append(gbest_fitness)

        best_solution.method = self.name
        return best_solution

    def _update_particle(
        self,
        position: List[int],
        pbest: List[int],
        gbest: List[int],
        n_jobs: int,
    ) -> List[int]:
        """
        Update particle position using order-based rules.

        For each position in the permutation:
          - With probability w: keep current job
          - With probability c1: move toward pbest ordering
          - With probability c2: move toward gbest ordering
        """
        # Build a new permutation via order crossover-like logic
        new_perm = [-1] * n_jobs
        used = set()
        jobs = set(range(n_jobs))

        # Copy some positions from position, pbest, or gbest
        for pos in range(n_jobs):
            r = self.rng.random()
            if r < self.w:
                candidate = position[pos]
            elif r < self.w + self.c1:
                candidate = pbest[pos]
            else:
                candidate = gbest[pos]

            if candidate not in used:
                new_perm[pos] = candidate
                used.add(candidate)

        # Fill remaining with unused jobs (in order of current position)
        remaining = [j for j in position if j not in used]
        idx = 0
        for pos in range(n_jobs):
            if new_perm[pos] == -1:
                new_perm[pos] = remaining[idx]
                idx += 1

        return new_perm
