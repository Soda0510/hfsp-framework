"""
ScheduleSolution: Data class holding a complete HFSP schedule.

Contains the job permutation, per-machine assignments, and computed
objective values (makespan, flow time, tardiness, energy).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .instance import HFSPInstance


@dataclass
class ScheduleSolution:
    """
    A complete HFSP schedule produced by decoding a job permutation.

    Attributes
    ----------
    permutation : list[int]
        Job processing order (job indices 0..n-1).
    instance : HFSPInstance
        Reference to the problem instance.
    assignments : list[dict]
        Each entry: {"job": int, "stage": int, "machine": int,
                      "start": float, "end": float}.
    makespan : float
        Completion time of the last job at the last stage (C_max).
    flow_time : float
        Total flow time (sum of completion times of all jobs).
    tardiness : float
        Total tardiness (requires due_dates on instance).
    energy : float
        Total energy consumption (requires power data on instance).
    weighted_objective : float
        Composite weighted objective value.
    """

    permutation: List[int]
    instance: "HFSPInstance"
    assignments: List[Dict[str, Any]] = field(default_factory=list)

    makespan: float = 0.0
    flow_time: float = 0.0
    tardiness: float = 0.0
    energy: float = 0.0
    weighted_objective: float = float("inf")

    # Pareto metadata (for multi-objective algorithms)
    rank: Optional[int] = None
    crowding_distance: Optional[float] = None

    # Metadata
    generation: int = 0
    method: str = ""

    def __post_init__(self):
        # Validate permutation contains valid job indices
        # (permutation may be partial for constructive heuristics like NEH)
        n = self.instance.num_jobs
        if self.permutation:
            assert len(self.permutation) <= n, (
                f"Permutation length ({len(self.permutation)}) > num_jobs ({n})."
            )
            for j in self.permutation:
                assert 0 <= j < n, f"Invalid job index {j} in permutation."
            assert len(set(self.permutation)) == len(self.permutation), (
                "Permutation contains duplicate job indices."
            )

    # ---- Computed properties ----

    @property
    def job_completion_times(self) -> np.ndarray:
        """Completion time of each job at the final stage (shape (n_jobs,))."""
        n = self.instance.num_jobs
        s_final = self.instance.num_stages - 1
        ct = np.zeros(n)
        for a in self.assignments:
            if a["stage"] == s_final:
                ct[a["job"]] = max(ct[a["job"]], a["end"])
        return ct

    @property
    def machine_schedules(self) -> Dict[int, List[Dict[str, Any]]]:
        """Group assignments by machine index."""
        schedules = {m: [] for m in range(self.instance.total_machines)}
        for a in self.assignments:
            schedules[a["machine"]].append(a)
        for m in schedules:
            schedules[m].sort(key=lambda a: a["start"])
        return schedules

    def copy(self) -> "ScheduleSolution":
        """Create a shallow copy (permutation list is copied)."""
        import copy
        return ScheduleSolution(
            permutation=list(self.permutation),
            instance=self.instance,
            assignments=list(self.assignments),
            makespan=self.makespan,
            flow_time=self.flow_time,
            tardiness=self.tardiness,
            energy=self.energy,
            weighted_objective=self.weighted_objective,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            generation=self.generation,
            method=self.method,
        )

    def __repr__(self) -> str:
        return (
            f"ScheduleSolution(makespan={self.makespan:.2f}, "
            f"flow_time={self.flow_time:.2f}, "
            f"perm={self.permutation[:5]}...)"
        )
