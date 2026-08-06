"""
Simulated Annealing for HFSP.

Uses NEH initialization + random mutation moves + Metropolis acceptance.
"""

from typing import Optional
import numpy as np
import time

from ..base import Method
from ...core.instance import HFSPInstance
from ...core.solution import ScheduleSolution
from ...core.decoder import ListSchedulingDecoder
from ..operators import SwapOperator, InsertOperator, InverseOperator
from ..heuristics import neh_heuristic


class SimulatedAnnealing(Method):
    """
    Simulated Annealing for HFSP.

    Parameters
    ----------
    initial_temperature : float
        Starting temperature.
    cooling_rate : float
        Multiplicative cooling factor (0..1), typically 0.95-0.99.
    max_iterations : int
        Maximum iterations per temperature level.
    max_total_iterations : int
        Hard stop on total iterations.
    decoder : ListSchedulingDecoder, optional
    rng : np.random.Generator, optional
    time_limit : float
    """

    name = "SA"

    def __init__(
        self,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.97,
        max_iterations: int = 300,
        max_total_iterations: int = 50000,
        decoder: ListSchedulingDecoder = None,
        rng: np.random.Generator = None,
        time_limit: float = float("inf"),
    ):
        super().__init__(rng=rng, time_limit=time_limit)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.max_total_iterations = max_total_iterations
        self.decoder = decoder if decoder is not None else ListSchedulingDecoder()
        self.mutation_ops = [SwapOperator(), InsertOperator(), InverseOperator()]

    def solve(self, instance: HFSPInstance) -> ScheduleSolution:
        self.start_time = time.perf_counter()
        self.convergence = []

        # Initialize with NEH solution
        current = neh_heuristic(instance, self.decoder, self.rng)
        best = current.copy()
        self.convergence.append(best.makespan)

        temperature = self.initial_temperature
        n_jobs = instance.num_jobs

        # Adaptive initial temperature (if too high for the problem scale)
        avg_pt = instance.sum_processing_times() / (instance.num_jobs * instance.total_machines)
        temperature = min(temperature, best.makespan * 0.1)

        total_iter = 0

        while total_iter < self.max_total_iterations:
            if self._check_time():
                break

            for _ in range(self.max_iterations):
                total_iter += 1

                # Generate neighbor
                op = self.mutation_ops[self.rng.integers(0, len(self.mutation_ops))]
                neighbor_perm = op.apply(current.permutation, self.rng)
                neighbor = self.decoder.decode(instance, neighbor_perm, self.rng)

                delta = neighbor.makespan - current.makespan

                # Metropolis acceptance
                if delta < 0 or self.rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                    current = neighbor
                    if current.makespan < best.makespan - 1e-12:
                        best = current.copy()

                self.convergence.append(best.makespan)

                if self._check_time():
                    break

            # Cool down
            temperature *= self.cooling_rate

            # Stop if temperature is too low
            if temperature < 1e-6:
                break

        best.method = self.name
        return best
