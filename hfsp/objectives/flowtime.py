"""
Total flow time objective: sum of completion times of all jobs.
"""

import numpy as np
from .base import ObjectiveFunction
from ..core.solution import ScheduleSolution


class FlowtimeObjective(ObjectiveFunction):
    """Total flow time = sum of completion times of all jobs at the final stage."""

    name = "flow_time"

    def compute(self, solution: ScheduleSolution) -> float:
        return solution.flow_time
