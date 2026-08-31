"""
ConfigurableMetaheuristic: executes an AlgorithmSpec as a ``Method``.

Three search templates are selected by the spec's structural switches:

- ``population_mode=True``            -> GA-style (tournament + crossover +
                                          mutation + elitism)
- ``population_mode=False`` and
  ``use_destroy_repair=True``         -> IG-style (destroy & reconstruct +
                                          acceptance criterion)
- ``population_mode=False`` and
  ``use_destroy_repair=False``        -> single-solution search (mutation +
                                          acceptance criterion)

If ``max_iterations <= 0`` the spec is treated as a pure constructive method
(initializer + optional local search), which lets NEH+LS be represented.

The RNG call sequence of the GA-style and IG-style templates deliberately
mirrors ``GeneticAlgorithm`` and ``IteratedGreedy`` so that seed specs
reproduce the hand-crafted implementations exactly (verified in
``scripts/verify_executor.py``).
"""

import time
from typing import List, Optional
import numpy as np

from ..core.instance import HFSPInstance
from ..core.solution import ScheduleSolution
from ..core.decoder import ListSchedulingDecoder
from ..methods.base import Method
from ..methods.operators import local_search
from .components import resolve_initializer, resolve_mutation, resolve_crossover, validate_spec
from .representation import AlgorithmSpec


class ConfigurableMetaheuristic(Method):
    """A metaheuristic assembled from an AlgorithmSpec."""

    name = "ConfigurableMetaheuristic"

    def __init__(
        self,
        spec: AlgorithmSpec,
        decoder: Optional[ListSchedulingDecoder] = None,
        rng: Optional[np.random.Generator] = None,
        time_limit: float = float("inf"),
    ):
        super().__init__(rng=rng, time_limit=time_limit)
        errors = validate_spec(spec)
        if errors:
            raise ValueError(f"Invalid AlgorithmSpec: {'; '.join(errors)}")
        self.spec = spec
        self.decoder = decoder if decoder is not None else ListSchedulingDecoder()
        # Pre-resolve operators (order matters: matches the registries).
        self.mutation_ops = [resolve_mutation(op) for op in spec.mutation_operators]
        self.crossover_ops = [resolve_crossover(op) for op in spec.crossover_operators]

    # ------------------------------------------------------------------
    # Public API (Method)
    # ------------------------------------------------------------------

    def solve(self, instance: HFSPInstance) -> ScheduleSolution:
        if self.spec.max_iterations <= 0:
            return self._solve_constructive(instance)
        if self.spec.population_mode:
            return self._solve_population(instance)
        if self.spec.use_destroy_repair:
            return self._solve_destroy_repair(instance)
        return self._solve_single(instance)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _initial_permutation(self, instance: HFSPInstance) -> List[int]:
        """Build the initial permutation from the spec's initializer."""
        fn = resolve_initializer(self.spec.initializer)
        if fn is not None:
            return fn(instance, self.decoder, self.rng).permutation
        perm = list(range(instance.num_jobs))
        self.rng.shuffle(perm)
        return perm

    def _accept(self, new_sol: ScheduleSolution,
                current_sol: ScheduleSolution, temperature: float,
                best_sol: Optional[ScheduleSolution] = None) -> bool:
        """Return True if new_sol should replace current_sol."""
        delta = new_sol.makespan - current_sol.makespan
        if delta < 0:
            return True
        spec = self.spec
        if spec.acceptance == "always":
            return True
        if spec.acceptance == "only_better":
            return False
        if spec.acceptance == "metropolis":
            return self.rng.random() < np.exp(-delta / max(temperature, 1e-12))
        if spec.acceptance == "threshold":
            th = spec.threshold if spec.threshold is not None else 0.0
            return delta <= th
        if spec.acceptance == "record_to_record":
            # Accept if within a small relative deviation of the record (best).
            r = spec.threshold if spec.threshold is not None else 0.01
            if best_sol is not None and best_sol.makespan > 0:
                return new_sol.makespan <= (1.0 + r) * best_sol.makespan
            return delta <= r * current_sol.makespan
        return False

    def _tournament(self, population: List[ScheduleSolution],
                    k: int) -> ScheduleSolution:
        """Tournament selection; RNG call order matches GeneticAlgorithm."""
        idxs = self.rng.integers(0, len(population), size=k)
        best = population[int(idxs[0])]
        for idx in idxs[1:]:
            if population[int(idx)].makespan < best.makespan:
                best = population[int(idx)]
        return best

    def _maybe_local_search(self, sol: ScheduleSolution) -> ScheduleSolution:
        spec = self.spec
        if spec.use_local_search:
            return local_search(
                sol, self.decoder,
                max_iterations=spec.local_search_iterations,
                strategy="first_improvement",
                rng=self.rng,
            )
        return sol

    # ------------------------------------------------------------------
    # Template 1: pure constructive (NEH / SPT / LPT [+ local search])
    # ------------------------------------------------------------------

    def _solve_constructive(self, instance: HFSPInstance) -> ScheduleSolution:
        self.start_time = time.perf_counter()
        self.convergence = []
        sol = self.decoder.decode(instance, self._initial_permutation(instance), self.rng)
        if self.spec.use_local_search:
            sol = self._maybe_local_search(sol)
        sol.method = self.spec.name
        self.convergence.append(sol.makespan)
        return sol

    # ------------------------------------------------------------------
    # Template 2: population-based (GA-style)
    # ------------------------------------------------------------------

    def _solve_population(self, instance: HFSPInstance) -> ScheduleSolution:
        self.start_time = time.perf_counter()
        self.convergence = []
        spec = self.spec
        pop_size = max(spec.population_size, spec.elite_size + 2)

        # Initial population: one initializer-seeded individual + random fill.
        population = []
        seed_perm = self._initial_permutation(instance)
        population.append(self.decoder.decode(instance, seed_perm, self.rng))
        for _ in range(pop_size - 1):
            perm = list(range(instance.num_jobs))
            self.rng.shuffle(perm)
            population.append(self.decoder.decode(instance, perm, self.rng))
        population.sort(key=lambda s: s.makespan)
        best = population[0].copy()
        self.convergence.append(best.makespan)

        for _gen in range(spec.max_iterations):
            if self._check_time():
                break

            new_pop = [population[i].copy()
                       for i in range(min(spec.elite_size, len(population)))]

            while len(new_pop) < pop_size:
                p1 = self._tournament(population, spec.tournament_size)
                p2 = self._tournament(population, spec.tournament_size)
                child_perm = list(p1.permutation)

                if spec.crossover_operators and self.rng.random() < spec.crossover_prob:
                    op = self.crossover_ops[int(self.rng.integers(0, len(self.crossover_ops)))]
                    c1, c2 = op.crossover(p1.permutation, p2.permutation, self.rng)
                    child_perm = c1 if self.rng.random() < 0.5 else c2

                if self.mutation_ops and self.rng.random() < spec.mutation_prob:
                    op = self.mutation_ops[int(self.rng.integers(0, len(self.mutation_ops)))]
                    child_perm = op.apply(child_perm, self.rng)

                new_pop.append(self.decoder.decode(instance, child_perm, self.rng))

            # Local search on the elite (matching GA's use_local_search semantics).
            if spec.use_local_search:
                for i in range(min(spec.elite_size, len(new_pop))):
                    new_pop[i] = self._maybe_local_search(new_pop[i])

            population = new_pop
            population.sort(key=lambda s: s.makespan)
            if population[0].makespan < best.makespan - 1e-12:
                best = population[0].copy()
            self.convergence.append(best.makespan)

        best.method = self.spec.name
        return best

    # ------------------------------------------------------------------
    # Template 3: destroy & reconstruct (IG-style)
    # ------------------------------------------------------------------

    def _solve_destroy_repair(self, instance: HFSPInstance) -> ScheduleSolution:
        self.start_time = time.perf_counter()
        self.convergence = []
        spec = self.spec
        n_jobs = instance.num_jobs

        d = spec.destruction_size
        if d is None:
            d = max(2, n_jobs // 10)

        current = self.decoder.decode(instance, self._initial_permutation(instance), self.rng)
        best = current.copy()
        self.convergence.append(best.makespan)

        # Adaptive temperature (matches IteratedGreedy when spec.temperature is None).
        if spec.temperature is not None:
            T = spec.temperature
        else:
            T = 0.05 * best.makespan / np.log(2) if best.makespan > 0 else 1.0

        for _ in range(spec.max_iterations):
            if self._check_time():
                break

            # Destruction: remove d random jobs.
            perm = list(current.permutation)
            removed = []
            for _ in range(d):
                if len(perm) <= 1:
                    break
                idx = int(self.rng.integers(0, len(perm)))
                removed.append(perm.pop(idx))

            # Construction: NEH-style reinsertion at best position.
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
            if spec.use_local_search:
                new_sol = self._maybe_local_search(new_sol)

            if self._accept(new_sol, current, T, best):
                current = new_sol
                if current.makespan < best.makespan - 1e-12:
                    best = current.copy()

            self.convergence.append(best.makespan)

        best.method = self.spec.name
        return best

    # ------------------------------------------------------------------
    # Template 4: single-solution search (SA/ILS-style)
    # ------------------------------------------------------------------

    def _solve_single(self, instance: HFSPInstance) -> ScheduleSolution:
        self.start_time = time.perf_counter()
        self.convergence = []
        spec = self.spec

        current = self.decoder.decode(instance, self._initial_permutation(instance), self.rng)
        best = current.copy()
        self.convergence.append(best.makespan)

        if spec.temperature is not None:
            T = spec.temperature
        else:
            T = min(100.0, best.makespan * 0.1) if best.makespan > 0 else 100.0

        for _ in range(spec.max_iterations):
            if self._check_time():
                break

            op = self.mutation_ops[int(self.rng.integers(0, len(self.mutation_ops)))]
            neighbor = self.decoder.decode(
                instance, op.apply(current.permutation, self.rng), self.rng)

            if self._accept(neighbor, current, T, best):
                current = neighbor
                if current.makespan < best.makespan - 1e-12:
                    best = current.copy()

            if spec.acceptance == "metropolis":
                T *= spec.cooling_rate

            self.convergence.append(best.makespan)

        best.method = self.spec.name
        return best
