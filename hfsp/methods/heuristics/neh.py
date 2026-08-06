"""
NEH heuristic adapted for Hybrid Flow Shop Scheduling.

The original NEH (Nawaz-Enscore-Ham, 1983) was designed for permutation
flow shop. This adaptation uses the list-scheduling decoder for HFSP:

1. Sort jobs by total processing time (descending).
2. Take the first job → initial partial sequence.
3. For each remaining job k:
   - Try inserting job k at every possible position in the current partial sequence.
   - Keep the position that yields the best makespan after decoding.
"""

from typing import List
import numpy as np

from ...core.instance import HFSPInstance
from ...core.solution import ScheduleSolution
from ...core.decoder import ListSchedulingDecoder


def neh_heuristic(
    instance: HFSPInstance,
    decoder: ListSchedulingDecoder = None,
    rng: np.random.Generator = None,
) -> ScheduleSolution:
    """
    NEH heuristic for HFSP.

    Parameters
    ----------
    instance : HFSPInstance
    decoder : ListSchedulingDecoder, optional
    rng : np.random.Generator, optional

    Returns
    -------
    ScheduleSolution
    """
    if decoder is None:
        decoder = ListSchedulingDecoder(tie_breaking="first")

    n_jobs = instance.num_jobs
    pt = instance.processing_times  # (n, m)

    # Step 1: Sort jobs by total processing time (descending)
    total_pt = np.sum(pt, axis=1)  # sum over all machines
    job_order = list(np.argsort(-total_pt))  # descending

    # Step 2: Initialize with the first job
    sequence = [job_order[0]]

    # Step 3: Insert remaining jobs one by one
    for k in range(1, n_jobs):
        job_k = job_order[k]
        best_pos = 0
        best_makespan = float("inf")

        # Try inserting at each position
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job_k] + sequence[pos:]
            sol = decoder.decode(instance, candidate, rng)
            if sol.makespan < best_makespan:
                best_makespan = sol.makespan
                best_pos = pos

        # Insert at best position
        sequence = sequence[:best_pos] + [job_k] + sequence[best_pos:]

    # Final decode with best sequence
    solution = decoder.decode(instance, sequence, rng)
    solution.method = "NEH"
    return solution
