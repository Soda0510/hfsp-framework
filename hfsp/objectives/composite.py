"""
Weighted sum composite objective.
"""

from .base import ObjectiveFunction
from .makespan import MakespanObjective
from .flowtime import FlowtimeObjective
from .tardiness import TardinessObjective
from ..core.solution import ScheduleSolution


class WeightedSumObjective(ObjectiveFunction):
    """
    Weighted sum of multiple objectives with normalization.

    obj = w_makespan * (C_max / ref_makespan)
        + w_flowtime * (F / ref_flowtime)
        + w_tardiness * (T / ref_tardiness)
    """

    name = "weighted_sum"

    def __init__(
        self,
        w_makespan: float = 1.0,
        w_flowtime: float = 0.0,
        w_tardiness: float = 0.0,
        ref_makespan: float = 1.0,
        ref_flowtime: float = 1.0,
        ref_tardiness: float = 1.0,
    ):
        self.w_makespan = w_makespan
        self.w_flowtime = w_flowtime
        self.w_tardiness = w_tardiness
        self.ref_makespan = ref_makespan
        self.ref_flowtime = ref_flowtime
        self.ref_tardiness = ref_tardiness

        self._ms_obj = MakespanObjective()
        self._ft_obj = FlowtimeObjective()
        self._td_obj = TardinessObjective()

    def compute(self, solution: ScheduleSolution) -> float:
        ms = self._ms_obj.compute(solution)
        ft = self._ft_obj.compute(solution)
        td = self._td_obj.compute(solution)

        value = (
            self.w_makespan * ms / self.ref_makespan
            + self.w_flowtime * ft / self.ref_flowtime
            + self.w_tardiness * td / self.ref_tardiness
        )
        return value
