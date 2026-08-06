"""
NSGA-II: Non-dominated Sorting Genetic Algorithm II (Deb et al., 2002).

Multi-objective evolutionary algorithm for permutation-based HFSP.
Optimizes two objectives simultaneously:
  - makespan (C_max)
  - total flow time (sum of completion times)

Key mechanisms:
  - Non-dominated sorting for rank assignment
  - Crowding distance for diversity preservation
  - Binary tournament selection (rank first, then crowding distance)
  - Elitism via parent + offspring merge and truncation
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
    crowding_distance,
    compute_objective_matrix,
    select_closest_to_ideal,
)


class NSGAII(Method):
    """
    NSGA-II for multi-objective HFSP.

    Parameters
    ----------
    population_size : int
        Number of individuals in the population.
    crossover_prob : float
        Probability of applying crossover to a pair of parents.
    mutation_prob : float
        Probability of applying mutation to each offspring.
    max_generations : int
        Maximum number of generations.
    tournament_size : int
        Tournament size for binary tournament selection.
    decoder : ListSchedulingDecoder, optional
        Decoder for permutation → schedule. Defaults to ListSchedulingDecoder().
    rng : np.random.Generator, optional
    time_limit : float
        Maximum wall-clock time in seconds.

    Attributes (post-solve)
    -----------------------
    pareto_front : list[ScheduleSolution]
        The final non-dominated front.
    """

    name = "NSGA-II"

    def __init__(
        self,
        population_size: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.3,
        max_generations: int = 500,
        tournament_size: int = 2,
        decoder: ListSchedulingDecoder = None,
        rng: np.random.Generator = None,
        time_limit: float = float("inf"),
    ):
        super().__init__(rng=rng, time_limit=time_limit)
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.decoder = decoder if decoder is not None else ListSchedulingDecoder()

        self.mutation_ops = [SwapOperator(), InsertOperator(), InverseOperator()]
        self.crossover_ops = [OrderCrossover(), PMXCrossover(), TwoPointCrossover()]

        self.pareto_front: List[ScheduleSolution] = []

    def solve(self, instance: HFSPInstance) -> ScheduleSolution:
        """
        Run NSGA-II on the given instance.

        Returns the knee point (closest to ideal) and stores the full
        Pareto front in self.pareto_front.
        """
        self.start_time = time.perf_counter()
        self.convergence = []
        self.pareto_front = []
        n_jobs = instance.num_jobs
        n_obj = 2  # makespan + flowtime

        # 1. Initialize population randomly
        population = self._init_population(instance, n_jobs)

        # 2. Evaluate and sort
        obj_matrix = compute_objective_matrix(population)
        fronts = non_dominated_sort(obj_matrix)

        # Assign ranks
        for front_idx, front in enumerate(fronts):
            for idx in front:
                population[idx].rank = front_idx

        # Compute crowding distance for each front
        for front in fronts:
            cd = crowding_distance(front, obj_matrix)
            for idx in front:
                population[idx].crowding_distance = cd[idx]

        # Record initial convergence (size of front 0)
        n_front0 = len(fronts[0]) if fronts else 0
        self.convergence.append(n_front0)

        # 3. Generation loop
        for gen in range(self.max_generations):
            if self._check_time():
                break

            # 3a. Create offspring
            offspring = self._create_offspring(population, instance, n_jobs, gen)

            # 3b. Merge parent + offspring (elitism)
            combined = population + offspring
            combined_obj = compute_objective_matrix(combined)

            # 3c. Non-dominated sort combined population
            fronts = non_dominated_sort(combined_obj)
            for front_idx, front in enumerate(fronts):
                for idx in front:
                    combined[idx].rank = front_idx

            # 3d. Truncate to population_size
            population = self._truncate(combined, fronts, combined_obj)

            # 3e. Record convergence
            final_obj = compute_objective_matrix(population)
            final_fronts = non_dominated_sort(final_obj)
            self.convergence.append(len(final_fronts[0]) if final_fronts else 0)

        # 4. Extract final Pareto front
        final_obj = compute_objective_matrix(population)
        final_fronts = non_dominated_sort(final_obj)
        self.pareto_front = [population[i].copy() for i in final_fronts[0]]
        for sol in self.pareto_front:
            sol.method = self.name

        # 5. Return knee point
        ideal = np.min(final_obj, axis=0)
        nadir = np.max(final_obj, axis=0)
        front_indices = final_fronts[0]
        front_solutions = [population[i] for i in front_indices]
        front_obj = final_obj[front_indices]

        best = select_closest_to_ideal(front_solutions, front_obj, ideal, nadir)
        best.method = self.name
        return best

    # ------- internal helpers -------

    def _init_population(self, instance: HFSPInstance, n_jobs: int) -> List[ScheduleSolution]:
        """Generate random initial population."""
        pop = []
        for _ in range(self.population_size):
            perm = list(range(n_jobs))
            self.rng.shuffle(perm)
            sol = self.decoder.decode(instance, perm, self.rng)
            pop.append(sol)
        return pop

    def _create_offspring(
        self,
        population: List[ScheduleSolution],
        instance: HFSPInstance,
        n_jobs: int,
        generation: int,
    ) -> List[ScheduleSolution]:
        """Create offspring via tournament selection, crossover, and mutation."""
        offspring = []
        while len(offspring) < self.population_size:
            # Binary tournament selection
            p1 = self._tournament_select(population)
            p2 = self._tournament_select(population)

            child_perm = list(p1.permutation)

            # Crossover
            if self.rng.random() < self.crossover_prob:
                op = self.crossover_ops[
                    self.rng.integers(0, len(self.crossover_ops))
                ]
                c1, c2 = op.crossover(p1.permutation, p2.permutation, self.rng)
                child_perm = c1 if self.rng.random() < 0.5 else c2

            # Mutation
            if self.rng.random() < self.mutation_prob:
                op = self.mutation_ops[
                    self.rng.integers(0, len(self.mutation_ops))
                ]
                child_perm = op.apply(child_perm, self.rng)

            child = self.decoder.decode(instance, child_perm, self.rng)
            child.generation = generation
            offspring.append(child)

        return offspring

    def _tournament_select(
        self, population: List[ScheduleSolution]
    ) -> ScheduleSolution:
        """Binary tournament: choose by rank, then crowding distance."""
        indices = self.rng.integers(0, len(population), size=self.tournament_size)
        best_idx = int(indices[0])
        for idx in indices[1:]:
            idx = int(idx)
            if self._crowded_comparison(population[idx], population[best_idx]):
                best_idx = idx
        return population[best_idx]

    @staticmethod
    def _crowded_comparison(a: ScheduleSolution, b: ScheduleSolution) -> bool:
        """
        Return True if 'a' is better than 'b'.

        Criterion: lower rank is better; if same rank, higher crowding
        distance is better (promotes diversity).
        """
        ra = a.rank if a.rank is not None else 999999
        rb = b.rank if b.rank is not None else 999999
        if ra < rb:
            return True
        if ra > rb:
            return False
        ca = a.crowding_distance if a.crowding_distance is not None else 0.0
        cb = b.crowding_distance if b.crowding_distance is not None else 0.0
        return ca > cb

    def _truncate(
        self,
        combined: List[ScheduleSolution],
        fronts: List[List[int]],
        obj_matrix: np.ndarray,
    ) -> List[ScheduleSolution]:
        """
        Fill new population front-by-front from the combined set.

        If the last front to fit exceeds the remaining capacity, sort its
        members by crowding distance and take the best.
        """
        new_pop = []
        for front in fronts:
            capacity = self.population_size - len(new_pop)
            if len(front) <= capacity:
                for idx in front:
                    new_pop.append(combined[idx])
            else:
                # Need to select a subset of this front
                cd = crowding_distance(front, obj_matrix)
                # Sort front members by crowding distance descending
                front_with_cd = [(idx, cd[idx]) for idx in front]
                front_with_cd.sort(key=lambda x: x[1], reverse=True)
                for idx, _ in front_with_cd[:capacity]:
                    new_pop.append(combined[idx])
                break

        return new_pop
