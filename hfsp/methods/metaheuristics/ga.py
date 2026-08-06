"""
Genetic Algorithm for HFSP.

Permutation-based encoding with tournament selection, elitism,
and multiple crossover/mutation operators.
"""

from typing import List, Optional
import numpy as np
import time
import copy

from ..base import Method
from ...core.instance import HFSPInstance
from ...core.solution import ScheduleSolution
from ...core.decoder import ListSchedulingDecoder
from ..operators import (
    SwapOperator, InsertOperator, InverseOperator,
    OrderCrossover, PMXCrossover, TwoPointCrossover,
    local_search,
)


class GeneticAlgorithm(Method):
    """
    Genetic Algorithm for HFSP.

    Parameters
    ----------
    population_size : int
        Number of individuals in the population.
    crossover_prob : float
        Probability of applying crossover (0..1).
    mutation_prob : float
        Probability of applying mutation (0..1).
    max_generations : int
        Maximum number of generations.
    elite_size : int
        Number of elite individuals to preserve.
    tournament_size : int
        Tournament size for selection.
    use_local_search : bool
        If True, apply local search to the best solution each generation.
    decoder : ListSchedulingDecoder, optional
    rng : np.random.Generator, optional
    time_limit : float
        Maximum runtime in seconds.
    """

    name = "GA"

    def __init__(
        self,
        population_size: int = 80,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.3,
        max_generations: int = 500,
        elite_size: int = 2,
        tournament_size: int = 2,
        use_local_search: bool = False,
        decoder: ListSchedulingDecoder = None,
        rng: np.random.Generator = None,
        time_limit: float = float("inf"),
    ):
        super().__init__(rng=rng, time_limit=time_limit)
        self.population_size = max(population_size, elite_size + 2)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_generations = max_generations
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.use_local_search = use_local_search
        self.decoder = decoder if decoder is not None else ListSchedulingDecoder()

        # Operators
        self.mutation_ops = [SwapOperator(), InsertOperator(), InverseOperator()]
        self.crossover_ops = [OrderCrossover(), PMXCrossover(), TwoPointCrossover()]

    def solve(self, instance: HFSPInstance) -> ScheduleSolution:
        self.start_time = time.perf_counter()
        self.convergence = []
        n_jobs = instance.num_jobs

        # Initialize population (random + NEH seed)
        population = []
        # Seed with some diverse solutions
        import numpy as np
        for _ in range(self.population_size):
            perm = list(range(n_jobs))
            self.rng.shuffle(perm)
            sol = self.decoder.decode(instance, perm, self.rng)
            population.append(sol)

        # Evaluate
        population.sort(key=lambda s: s.makespan)
        best = population[0].copy()
        self.convergence.append(best.makespan)

        # Generations
        for gen in range(self.max_generations):
            if self._check_time():
                break

            new_population = []

            # Elitism
            for i in range(self.elite_size):
                new_population.append(population[i].copy())

            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection (tournament)
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                child_perm = list(parent1.permutation)

                # Crossover
                if self.rng.random() < self.crossover_prob:
                    op = self.crossover_ops[self.rng.integers(0, len(self.crossover_ops))]
                    c1, c2 = op.crossover(parent1.permutation, parent2.permutation, self.rng)
                    child_perm = c1 if self.rng.random() < 0.5 else c2

                # Mutation
                if self.rng.random() < self.mutation_prob:
                    op = self.mutation_ops[self.rng.integers(0, len(self.mutation_ops))]
                    child_perm = op.apply(child_perm, self.rng)

                child = self.decoder.decode(instance, child_perm, self.rng)
                new_population.append(child)

            # Local search on elite
            if self.use_local_search:
                for i in range(self.elite_size):
                    new_population[i] = local_search(
                        new_population[i], self.decoder,
                        max_iterations=100, strategy="first_improvement",
                        rng=self.rng
                    )

            population = new_population
            population.sort(key=lambda s: s.makespan)

            if population[0].makespan < best.makespan - 1e-12:
                best = population[0].copy()
                best.generation = gen

            self.convergence.append(best.makespan)

        best.method = self.name
        return best

    def _tournament_select(self, population: List[ScheduleSolution]) -> ScheduleSolution:
        """Tournament selection: pick the best among k randomly chosen individuals."""
        candidates_idx = self.rng.integers(0, len(population), size=self.tournament_size)
        best_idx = candidates_idx[0]
        best_obj = population[best_idx].makespan
        for idx in candidates_idx[1:]:
            if population[idx].makespan < best_obj:
                best_obj = population[idx].makespan
                best_idx = idx
        return population[best_idx]
