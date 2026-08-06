"""
SPT (Shortest Processing Time) heuristic.

Sort jobs by total processing time ascending and decode.
"""

import numpy as np

from ...core.instance import HFSPInstance
from ...core.solution import ScheduleSolution
from ...core.decoder import ListSchedulingDecoder


def spt_heuristic(
    instance: HFSPInstance,
    decoder: ListSchedulingDecoder = None,
    rng: np.random.Generator = None,
) -> ScheduleSolution:
    """
    SPT heuristic: sort jobs by total processing time (ascending).

    Returns
    -------
    ScheduleSolution
    """
    if decoder is None:
        decoder = ListSchedulingDecoder()

    pt = instance.processing_times
    total_pt = np.sum(pt, axis=1)
    sequence = list(np.argsort(total_pt))  # ascending

    solution = decoder.decode(instance, sequence, rng)
    solution.method = "SPT"
    return solution
