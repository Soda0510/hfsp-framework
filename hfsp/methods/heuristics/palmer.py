"""
Palmer's slope-index heuristic adapted for HFSP.

Palmer (1965) sorts jobs by a slope index that reflects whether a job should
go early or late in the sequence.  Adapted here: the index uses the stage-level
processing times, and the resulting order is decoded by the HFSP decoder.
"""

from typing import List
import numpy as np

from ...core.instance import HFSPInstance
from ...core.solution import ScheduleSolution
from ...core.decoder import ListSchedulingDecoder


def _stage_pt(instance: HFSPInstance) -> np.ndarray:
    """Return stage-level processing times, shape (n, s)."""
    if instance.base_processing_times is not None and \
            instance.base_processing_times.shape == (instance.num_jobs, instance.num_stages):
        return np.asarray(instance.base_processing_times, dtype=float)
    n, s = instance.num_jobs, instance.num_stages
    pt = np.zeros((n, s))
    for st in range(s):
        m = int(instance.machines_in_stage(st)[0])
        pt[:, st] = instance.processing_times[:, m]
    return pt


def palmer_heuristic(
    instance: HFSPInstance,
    decoder: ListSchedulingDecoder = None,
    rng: np.random.Generator = None,
) -> ScheduleSolution:
    """Sort jobs by Palmer's slope index, decode, and return the schedule."""
    if decoder is None:
        decoder = ListSchedulingDecoder(tie_breaking="first")
    n, s = instance.num_jobs, instance.num_stages
    pt = _stage_pt(instance)

    def slope(j: int) -> float:
        # weight later stages higher; classic Palmer index.
        return float(sum((s - (2 * k + 1)) * pt[j, k] for k in range(s)))

    order = sorted(range(n), key=slope, reverse=True)
    sol = decoder.decode(instance, order, rng)
    sol.method = "PALMER"
    return sol
