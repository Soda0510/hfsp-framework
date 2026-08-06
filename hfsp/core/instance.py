"""
HFSPInstance: Data class representing a Hybrid Flow Shop Scheduling Problem instance.

Key attributes:
- num_jobs (n): number of jobs
- num_stages (s): number of stages in series
- machines_per_stage: list of machine counts per stage
- total_machines: total number of machines across all stages
- processing_times: (n × total_machines) array of processing times per job per machine
- machine_stage_idx: mapping from global machine index to stage index
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class HFSPInstance:
    """A Hybrid Flow Shop Scheduling Problem instance."""

    name: str
    num_jobs: int
    num_stages: int
    machines_per_stage: list[int]   # length = num_stages
    total_machines: int

    # Core data: processing time of each job on each machine
    # Shape: (num_jobs, total_machines)
    processing_times: np.ndarray

    # Derived: which stage each global machine index belongs to (0-indexed)
    machine_stage_idx: np.ndarray = field(init=False)

    # ---- Optional extended fields (energy-aware, due-date-aware) ----
    base_processing_times: Optional[np.ndarray] = None   # (n, s) original PT per stage
    speed: Optional[np.ndarray] = None                    # (total_machines,)
    conversion_factor: Optional[np.ndarray] = None        # (total_machines,)
    power_on: Optional[np.ndarray] = None                 # (total_machines,)
    power_idle: Optional[np.ndarray] = None               # (total_machines,)
    power_reset: Optional[np.ndarray] = None              # (total_machines,)
    break_even_point: Optional[np.ndarray] = None         # (total_machines,)
    due_dates: Optional[np.ndarray] = None                # (num_jobs,)
    tao: Optional[float] = None                           # energy weight
    R: Optional[float] = None                             # tardiness weight

    # Q-tables (pre-loaded from data; used by Q-learning algorithms)
    q_table: Optional[np.ndarray] = None      # (6, 6)
    q_table_1: Optional[np.ndarray] = None    # (9, 9)
    q_table_2: Optional[np.ndarray] = None    # (6, 6)

    def __post_init__(self):
        """Validate consistency and compute derived fields."""
        # Validate shapes
        n, m = self.processing_times.shape
        assert n == self.num_jobs, (
            f"Processing times rows ({n}) != num_jobs ({self.num_jobs})."
        )
        assert m == self.total_machines, (
            f"Processing times cols ({m}) != total_machines ({self.total_machines})."
        )
        assert sum(self.machines_per_stage) == self.total_machines, (
            f"Sum of machines_per_stage ({sum(self.machines_per_stage)}) "
            f"!= total_machines ({self.total_machines})."
        )
        assert len(self.machines_per_stage) == self.num_stages, (
            f"Length of machines_per_stage ({len(self.machines_per_stage)}) "
            f"!= num_stages ({self.num_stages})."
        )

        # Build machine_stage_idx
        stages = []
        for s, count in enumerate(self.machines_per_stage):
            stages.extend([s] * count)
        self.machine_stage_idx = np.array(stages, dtype=int)

    # ---- Convenience methods ----

    def machines_in_stage(self, stage: int) -> np.ndarray:
        """Return global machine indices belonging to a given stage (0-indexed)."""
        return np.where(self.machine_stage_idx == stage)[0]

    def stage_of_machine(self, machine: int) -> int:
        """Return the stage index (0-indexed) for a global machine index."""
        return int(self.machine_stage_idx[machine])

    def pt(self, job: int, machine: int) -> float:
        """Processing time of job j on machine m."""
        return float(self.processing_times[job, machine])

    def sum_processing_times(self) -> float:
        """Sum of all processing times (used for normalization)."""
        return float(np.sum(self.processing_times))

    @property
    def total_machines_stage(self) -> int:
        """Convenience: return max machines in any stage (assumes uniform)."""
        return max(self.machines_per_stage) if self.machines_per_stage else 0

    def __repr__(self) -> str:
        return (
            f"HFSPInstance(name={self.name!r}, jobs={self.num_jobs}, "
            f"stages={self.num_stages}, machines={self.total_machines}, "
            f"m_per_stage={self.machines_per_stage})"
        )
