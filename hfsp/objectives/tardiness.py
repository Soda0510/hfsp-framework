"""
Total tardiness objective: sum of max(0, C_j - d_j).
"""

import numpy as np
from .base import ObjectiveFunction
from ..core.solution import ScheduleSolution


class TardinessObjective(ObjectiveFunction):
    """Total tardiness = sum(max(0, C_j - d_j))."""

    name = "tardiness"

    def compute(self, solution: ScheduleSolution) -> float:
        instance = solution.instance
        if instance.due_dates is None:
            return 0.0
        completion_times = solution.job_completion_times
        tardiness = np.maximum(0, completion_times - instance.due_dates)
        return float(np.sum(tardiness))
