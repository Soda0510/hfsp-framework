"""
Makespan objective: C_max = max_j C_j
"""

import numpy as np
from .base import ObjectiveFunction
from ..core.solution import ScheduleSolution


class MakespanObjective(ObjectiveFunction):
    """Makespan = maximum completion time among all jobs."""

    name = "makespan"

    def compute(self, solution: ScheduleSolution) -> float:
        return solution.makespan

    @staticmethod
    def compute_from_assignments(instance: "HFSPInstance",
                                  assignments: list) -> float:
        """Compute makespan directly from assignments (avoids full solution construction)."""
        n_stages = instance.num_stages
        final_stage = n_stages - 1
        max_ct = 0.0
        for a in assignments:
            if a["stage"] == final_stage:
                if a["end"] > max_ct:
                    max_ct = a["end"]
        return max_ct
