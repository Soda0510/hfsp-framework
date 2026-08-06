"""
Local search framework for HFSP permutation solutions.

Supports:
- first_improvement: accept the first move that improves the objective.
- best_improvement: evaluate all moves and pick the best.
"""

from typing import List, Callable, Optional
import numpy as np

from ...core.instance import HFSPInstance
from ...core.solution import ScheduleSolution
from ...core.decoder import ListSchedulingDecoder


def local_search(
    solution: ScheduleSolution,
    decoder: ListSchedulingDecoder,
    max_iterations: int = 1000,
    strategy: str = "first_improvement",
    operators: Optional[List[Callable]] = None,
    rng: Optional[np.random.Generator] = None,
) -> ScheduleSolution:
    """
    Apply local search to improve a solution.

    Parameters
    ----------
    solution : ScheduleSolution
        Initial solution.
    decoder : ListSchedulingDecoder
        Decoder for evaluating neighbor solutions.
    max_iterations : int
        Maximum iterations without improvement.
    strategy : str
        "first_improvement" or "best_improvement".
    operators : list[Callable], optional
        List of operator functions (permutation -> new permutation).
        Default: [swap, insert, inverse].
    rng : np.random.Generator, optional

    Returns
    -------
    ScheduleSolution
        Best solution found.
    """
    if operators is None:
        from .swap import SwapOperator
        from .insert import InsertOperator
        from .inverse import InverseOperator
        operators = [
            SwapOperator().apply,
            InsertOperator().apply,
            InverseOperator().apply,
        ]

    if rng is None:
        rng = np.random.default_rng()

    current = solution
    best = solution
    best_obj = best.makespan
    n_jobs = solution.instance.num_jobs

    no_improve_count = 0

    while no_improve_count < max_iterations:
        improved = False

        if strategy == "first_improvement":
            # Randomly pick an operator and apply it
            op = operators[int(rng.integers(0, len(operators)))]
            neighbor_perm = op(current.permutation, rng)

            neighbor_sol = decoder.decode(current.instance, neighbor_perm, rng)

            if neighbor_sol.makespan < best_obj - 1e-12:
                best_obj = neighbor_sol.makespan
                best = neighbor_sol
                current = neighbor_sol
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1

        elif strategy == "best_improvement":
            # Try insert at all positions (most effective move)
            best_neighbor = None
            best_neighbor_obj = float("inf")

            for op in operators:
                for _ in range(min(n_jobs * 2, 200)):
                    neighbor_perm = op(current.permutation, rng)
                    neighbor_sol = decoder.decode(current.instance, neighbor_perm, rng)

                    if neighbor_sol.makespan < best_neighbor_obj - 1e-12:
                        best_neighbor_obj = neighbor_sol.makespan
                        best_neighbor = neighbor_sol

            if best_neighbor is not None and best_neighbor_obj < best_obj - 1e-12:
                best_obj = best_neighbor_obj
                best = best_neighbor
                current = best_neighbor
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += n_jobs * 2

    best.method = f"LS-{strategy}"
    return best
