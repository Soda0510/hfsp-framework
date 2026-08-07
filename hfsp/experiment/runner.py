"""
Batch experiment runner.

Orchestrates multi-instance, multi-algorithm, multi-run experiments.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

from ..core.instance import HFSPInstance
from ..io.instance_reader import InstanceReader
from ..utils.random import RNGManager
from ..methods.heuristics import neh_heuristic, spt_heuristic, lpt_heuristic
from ..methods.metaheuristics import GeneticAlgorithm, SimulatedAnnealing, IteratedGreedy, DiscretePSO
from ..methods.metaheuristics import NSGAII, MOEAD
from .config import ExperimentConfig, AlgorithmConfig
from .statistics import RunResult, ExperimentResultSet


class ExperimentRunner:
    """
    Runs batch experiments.

    Usage:
        config = ExperimentConfig(...)
        reader = InstanceReader("Data")
        runner = ExperimentRunner(config, reader)
        results = runner.run()
        results.save_csv("results/summary.csv")
    """

    def __init__(self, config: ExperimentConfig, reader: InstanceReader = None):
        self.config = config
        self.reader = reader if reader is not None else InstanceReader("Data")
        self.rng_mgr = RNGManager(base_seed=config.seed_base)

    def run(self) -> ExperimentResultSet:
        """Run the full experiment and return results."""
        all_results = ExperimentResultSet()

        # Discover instances
        instance_names = self._resolve_instances()

        # For progress reporting
        total_runs = len(instance_names) * len(self.config.algorithms) * self.config.num_runs

        with tqdm(total=total_runs, desc="Experiment", unit="run") as pbar:
            for inst_name in instance_names:
                instance = self.reader.load(inst_name)

                for algo_config in self.config.algorithms:
                    for run_id in range(self.config.num_runs):
                        try:
                            result = self._run_single(
                                instance, algo_config, run_id
                            )
                            all_results.results.append(result)
                        except Exception as e:
                            tqdm.write(f"ERROR: {inst_name}/{algo_config.name}/run{run_id}: {e}")
                        pbar.update(1)

        return all_results

    def _run_single(
        self, instance: HFSPInstance, algo_config: AlgorithmConfig, run_id: int
    ) -> RunResult:
        """Run a single algorithm on a single instance."""
        rng = self.rng_mgr.get_generator(run_id=run_id, stream=algo_config.name)

        t_start = time.perf_counter()

        # Build algorithm
        method = self._build_method(algo_config, rng)

        # Solve
        solution = method.solve(instance)

        runtime = time.perf_counter() - t_start

        return RunResult(
            instance_name=instance.name,
            algorithm_name=algo_config.name,
            run_id=run_id,
            makespan=solution.makespan,
            flow_time=solution.flow_time,
            tardiness=solution.tardiness,
            energy=solution.energy,
            weighted_obj=solution.weighted_objective,
            runtime=runtime,
            convergence=method.convergence if self.config.save_convergence else None,
        )

    def _build_method(self, algo_config: AlgorithmConfig, rng: np.random.Generator):
        """Factory method: build algorithm from config."""
        name = algo_config.name.upper()

        if name == "NEH":
            # NEH is stateless; wrap in a simple adapter
            from ..core.decoder import ListSchedulingDecoder
            decoder = ListSchedulingDecoder()
            class NEHAdapter:
                convergence = []
                def solve(self, inst):
                    sol = neh_heuristic(inst, decoder, rng)
                    self.convergence.append(sol.makespan)
                    return sol
            return NEHAdapter()

        elif name == "SPT":
            from ..core.decoder import ListSchedulingDecoder
            decoder = ListSchedulingDecoder()
            class SPTAdapter:
                convergence = []
                def solve(self, inst):
                    sol = spt_heuristic(inst, decoder, rng)
                    self.convergence.append(sol.makespan)
                    return sol
            return SPTAdapter()

        elif name == "LPT":
            from ..core.decoder import ListSchedulingDecoder
            decoder = ListSchedulingDecoder()
            class LPTAdapter:
                convergence = []
                def solve(self, inst):
                    sol = lpt_heuristic(inst, decoder, rng)
                    self.convergence.append(sol.makespan)
                    return sol
            return LPTAdapter()

        elif name == "GA":
            return GeneticAlgorithm(
                population_size=algo_config.population_size,
                crossover_prob=algo_config.crossover_prob,
                mutation_prob=algo_config.mutation_prob,
                max_generations=algo_config.max_generations,
                elite_size=algo_config.elite_size,
                rng=rng,
                time_limit=algo_config.time_limit,
            )

        elif name == "SA":
            return SimulatedAnnealing(
                initial_temperature=algo_config.initial_temperature,
                cooling_rate=algo_config.cooling_rate,
                max_total_iterations=algo_config.max_total_iterations,
                rng=rng,
                time_limit=algo_config.time_limit,
            )

        elif name == "IG":
            return IteratedGreedy(
                max_iterations=algo_config.ig_iterations,
                use_local_search=algo_config.ig_use_local_search,
                rng=rng,
                time_limit=algo_config.time_limit,
            )

        elif name == "DPSO":
            return DiscretePSO(
                swarm_size=algo_config.population_size,
                max_iterations=algo_config.max_generations,
                rng=rng,
                time_limit=algo_config.time_limit,
            )

        elif name in ("NSGA-II", "NSGA2"):
            return NSGAII(
                population_size=algo_config.population_size,
                crossover_prob=algo_config.crossover_prob,
                mutation_prob=algo_config.mutation_prob,
                max_generations=algo_config.max_generations,
                tournament_size=algo_config.tournament_size,
                rng=rng,
                time_limit=algo_config.time_limit,
            )

        elif name in ("MOEA/D", "MOEAD"):
            return MOEAD(
                population_size=algo_config.population_size,
                H=algo_config.moead_H,
                T=algo_config.moead_T,
                delta=algo_config.moead_delta,
                nr=algo_config.moead_nr,
                crossover_prob=algo_config.crossover_prob,
                mutation_prob=algo_config.mutation_prob,
                max_generations=algo_config.max_generations,
                rng=rng,
                time_limit=algo_config.time_limit,
            )

        else:
            raise ValueError(f"Unknown algorithm: {algo_config.name}")

    def _resolve_instances(self) -> List[str]:
        """Resolve instance names from config patterns."""
        if self.config.instances:
            # Filter by specified names (supports exact names only for now)
            available = set(self.reader.list_instances())
            requested = set(self.config.instances)
            # Also support wildcard like "10-*"
            if any("*" in p for p in self.config.instances):
                import fnmatch
                resolved = []
                for pattern in self.config.instances:
                    resolved.extend(fnmatch.filter(available, pattern))
                return sorted(resolved, key=self._sort_key)
            return sorted(requested & available, key=self._sort_key)
        return self.reader.list_instances()

    @staticmethod
    def _sort_key(name: str) -> tuple:
        parts = name.split("-")
        return tuple(int(p) for p in parts)
