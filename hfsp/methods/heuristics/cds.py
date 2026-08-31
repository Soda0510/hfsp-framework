"""
CDS (Campbell–Dudek–Smith) heuristic adapted for HFSP.

CDS (1970) is a multi-pass extension of Johnson's two-machine rule for flow
shops: it builds s-1 sub-problems, applies Johnson's rule to each, and keeps
the best sequence.  Adapted here for HFSP: each candidate order is decoded by
the HFSP decoder and the best makespan wins.
"""

from typing import List
import numpy as np

from ...core.instance import HFSPInstance
from ...core.solution import ScheduleSolution
from ...core.decoder import ListSchedulingDecoder


def _stage_pt(instance: HFSPInstance) -> np.ndarray:
    if instance.base_processing_times is not None and \
            instance.base_processing_times.shape == (instance.num_jobs, instance.num_stages):
        return np.asarray(instance.base_processing_times, dtype=float)
    n, s = instance.num_jobs, instance.num_stages
    pt = np.zeros((n, s))
    for st in range(s):
        m = int(instance.machines_in_stage(st)[0])
        pt[:, st] = instance.processing_times[:, m]
    return pt


def _johnson(a: List[float], b: List[float]) -> List[int]:
    """Johnson's rule for two machines given per-job times (a, b)."""
    jobs = list(range(len(a)))
    first = sorted([j for j in jobs if a[j] <= b[j]], key=lambda j: a[j])
    second = sorted([j for j in jobs if a[j] > b[j]], key=lambda j: b[j], reverse=True)
    return first + second


def cds_heuristic(
    instance: HFSPInstance,
    decoder: ListSchedulingDecoder = None,
    rng: np.random.Generator = None,
) -> ScheduleSolution:
    """Run CDS multi-pass and return the best schedule found."""
    if decoder is None:
        decoder = ListSchedulingDecoder(tie_breaking="first")
    n, s = instance.num_jobs, instance.num_stages
    pt = _stage_pt(instance)

    best_sol = None
    for k in range(1, s):
        a = [float(np.sum(pt[j, :k])) for j in range(n)]
        b = [float(np.sum(pt[j, s - k:])) for j in range(n)]
        order = _johnson(a, b)
        sol = decoder.decode(instance, order, rng)
        if best_sol is None or sol.makespan < best_sol.makespan:
            best_sol = sol

    if best_sol is None:
        best_sol = decoder.decode(instance, list(range(n)), rng)
    best_sol.method = "CDS"
    return best_sol
