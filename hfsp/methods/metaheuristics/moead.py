"""
MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition
(Zhang & Li, 2007).

Decomposes the MOP into N scalar subproblems using Tchebycheff
aggregation. Each subproblem has its own weight vector and neighbors.
Neighboring subproblems share information through limited neighborhood
updates.

Key differences from NSGA-II:
  - No non-dominated sorting during evolution
  - Uses weight vectors to scalarize objectives
  - Neighborhood-based local competition
  - External population (EP) archives the Pareto front
"""

from typing import List, Optional
import numpy as np
import time

from ..base import Method
from ...core.instance import HFSPInstance
from ...core.solution import ScheduleSolution
from ...core.decoder import ListSchedulingDecoder
from ..operators import (
    SwapOperator, InsertOperator, InverseOperator,
    OrderCrossover, PMXCrossover, TwoPointCrossover,
)
from .pareto import (
    non_dominated_sort,
    generate_weight_vectors,
    tchebycheff_decomposition,
    compute_objective_matrix,
    select_closest_to_ideal,
)


class MOEAD(Method):
    """
    MOEA/D for multi-objective HFSP.

    Parameters
    ----------
    population_size : int
        Number of subproblems (= number of weight vectors).
    H : int
        Number of divisions for Das & Dennis weight vector generation.
        For 2 objectives: pop_size ≈ H + 1.
    T : int
        Neighborhood size. Typically pop_size // 10.
    delta : float
        Probability of selecting mating parents from the neighborhood
        (vs. from the whole population). Higher = more local search.
    nr : int
        Maximum number of neighboring solutions a child can replace.
    crossover_prob : float
    mutation_prob : float
    max_generations : int
    decoder : ListSchedulingDecoder, optional
    rng : np.random.Generator, optional
    time_limit : float

    Attributes (post-solve)
    -----------------------
    pareto_front : list[ScheduleSolution]
        The final non-dominated front from the external population.
    """

    name = "MOEA/D"

    def __init__(
        self,
        population_size: int = 100,
        H: int = 99,
        T: int = 20,
        delta: float = 0.9,
        nr: int = 2,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.3,
        max_generations: int = 500,
        decoder: ListSchedulingDecoder = None,
        rng: np.random.Generator = None,
        time_limit: float = float("inf"),
    ):
        super().__init__(rng=rng, time_limit=time_limit)
        self.population_size = population_size
        self.H = H
        self.T = T
        self.delta = delta
        self.nr = nr
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_generations = max_generations
        self.decoder = decoder if decoder is not None else ListSchedulingDecoder()

        self.mutation_ops = [SwapOperator(), InsertOperator(), InverseOperator()]
        self.crossover_ops = [OrderCrossover(), PMXCrossover(), TwoPointCrossover()]

        self.pareto_front: List[ScheduleSolution] = []

    def solve(self, instance: HFSPInstance) -> ScheduleSolution:
        """
        Run MOEA/D on the given instance.

        Returns the knee point and stores the full Pareto front
        in self.pareto_front.
        """
        self.start_time = time.perf_counter()
        self.convergence = []
        self.pareto_front = []
        n_jobs = instance.num_jobs
        n_obj = 2

        # 1. Generate weight vectors
        weight_vectors = self._resolve_weight_vectors(n_obj)

        # 2. Build neighborhood indices by Euclidean distance in weight space
        B = self._build_neighborhood(weight_vectors)  # (N, T)

        # 3. Initialize population (one solution per subproblem)
        population = self._init_population(instance, n_jobs)
        obj_matrix = compute_objective_matrix(population)

        # 4. Initialize ideal point z*
        ideal_point = np.min(obj_matrix, axis=0).copy()

        # 5. External population (EP) — stores non-dominated solutions found
        ep = [sol.copy() for sol in population]

        # Record initial EP size
        self.convergence.append(len(ep))

        # 6. Main loop
        for gen in range(self.max_generations):
            if self._check_time():
                break

            for i in range(self.population_size):
                # 6a. Select parents: from neighborhood or whole population
                if self.rng.random() < self.delta:
                    pool = B[i]
                else:
                    pool = np.arange(self.population_size)

                p1_idx, p2_idx = self.rng.choice(pool, size=2, replace=False)
                p1 = population[int(p1_idx)]
                p2 = population[int(p2_idx)]

                # 6b. Crossover + Mutation
                child_perm = list(p1.permutation)
                if self.rng.random() < self.crossover_prob:
                    op = self.crossover_ops[
                        self.rng.integers(0, len(self.crossover_ops))
                    ]
                    c1, c2 = op.crossover(p1.permutation, p2.permutation, self.rng)
                    child_perm = c1 if self.rng.random() < 0.5 else c2

                if self.rng.random() < self.mutation_prob:
                    op = self.mutation_ops[
                        self.rng.integers(0, len(self.mutation_ops))
                    ]
                    child_perm = op.apply(child_perm, self.rng)

                child = self.decoder.decode(instance, child_perm, self.rng)
                child_obj = np.array([child.makespan, child.flow_time])

                # 6c. Update ideal point
                ideal_point = np.minimum(ideal_point, child_obj)

                # 6d. Update neighboring solutions
                n_replace = 0
                shuffled_B = list(B[i])
                self.rng.shuffle(shuffled_B)
                for j in shuffled_B:
                    j = int(j)
                    child_scalar = tchebycheff_decomposition(
                        child_obj, weight_vectors[j], ideal_point
                    )
                    current_obj = np.array([
                        population[j].makespan,
                        population[j].flow_time,
                    ])
                    current_scalar = tchebycheff_decomposition(
                        current_obj, weight_vectors[j], ideal_point
                    )

                    if child_scalar < current_scalar - 1e-12:
                        population[j] = child.copy()
                        n_replace += 1
                        if n_replace >= self.nr:
                            break

                # 6e. Update external population
                ep = self._update_ep(ep, child)

            # 7. Record convergence (EP size)
            self.convergence.append(len(ep))

        # 8. Final Pareto front from EP
        ep_obj = compute_objective_matrix(ep)
        ep_fronts = non_dominated_sort(ep_obj)
        self.pareto_front = [ep[i].copy() for i in ep_fronts[0]]
        for sol in self.pareto_front:
            sol.method = self.name

        # 9. Return knee point
        front_indices = ep_fronts[0]
        front_solutions = [ep[i] for i in front_indices]
        front_obj = ep_obj[front_indices]

        best = select_closest_to_ideal(
            front_solutions, front_obj,
            ideal_point, np.max(ep_obj, axis=0),
        )
        best.method = self.name
        return best

    # ------- internal helpers -------

    def _resolve_weight_vectors(self, n_obj: int) -> np.ndarray:
        """Generate weight vectors, matching population_size."""
        wv = generate_weight_vectors(n_obj, self.H, self.rng)
        N = len(wv)

        if self.population_size < N:
            # Subsample
            idxs = self.rng.choice(N, self.population_size, replace=False)
            wv = wv[idxs]
        elif self.population_size > N:
            # Pad with random normalized vectors
            extra = self.population_size - N
            rand_w = self.rng.random((extra, n_obj))
            rand_w = rand_w / rand_w.sum(axis=1, keepdims=True)
            wv = np.vstack([wv, rand_w])
            # Update population size to match
            self.population_size = len(wv)

        return wv

    def _build_neighborhood(self, weight_vectors: np.ndarray) -> np.ndarray:
        """
        Build T-nearest-neighbor index for each weight vector
        based on Euclidean distance.
        """
        N = len(weight_vectors)
        T = min(self.T, N)
        # Pairwise Euclidean distances
        dist = np.zeros((N, N))
        for i in range(N):
            dist[i] = np.sqrt(np.sum((weight_vectors - weight_vectors[i]) ** 2, axis=1))
        B = np.argsort(dist, axis=1)[:, :T]  # (N, T)
        return B

    def _init_population(
        self, instance: HFSPInstance, n_jobs: int
    ) -> List[ScheduleSolution]:
        """Random initialization."""
        pop = []
        for _ in range(self.population_size):
            perm = list(range(n_jobs))
            self.rng.shuffle(perm)
            sol = self.decoder.decode(instance, perm, self.rng)
            pop.append(sol)
        return pop

    def _update_ep(
        self, ep: List[ScheduleSolution], child: ScheduleSolution
    ) -> List[ScheduleSolution]:
        """
        Insert child into external population.
        Remove any solutions dominated by child.
        Do not insert if child is dominated by any EP member.
        """
        child_obj = np.array([child.makespan, child.flow_time])
        new_ep = []
        dominated = False

        for sol in ep:
            sol_obj = np.array([sol.makespan, sol.flow_time])
            if self._dominates(sol_obj, child_obj):
                # An EP member dominates the child — do not insert
                dominated = True
                break
            elif not self._dominates(child_obj, sol_obj):
                # Mutually non-dominated — keep EP member
                new_ep.append(sol)
            # else: child dominates this EP member — drop it

        if not dominated:
            new_ep.append(child.copy())

        return new_ep

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """Return True if a dominates b (minimization)."""
        return bool(np.all(a <= b) and np.any(a < b))
