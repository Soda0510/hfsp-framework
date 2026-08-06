"""
Energy consumption objective for HFSP with machine turn-off decisions.

Energy model:
- When processing: consumes power_on[m] per time unit
- When idle (on but not processing): consumes power_idle[m] per time unit
- When turned off and restarted: consumes power_reset[m] per reset event
- Break-even point: if idle time > break_even_point[m], it's beneficial to turn off

The algorithm scans each machine's timeline, detecting idle gaps.
For each idle gap:
  - If gap >= break_even_point[m]: turn off → cost = break_even_point[m] * power_idle[m] + power_reset[m]
    (wait break_even time, then turn off, pay reset cost when restarting)
  - If gap < break_even_point[m]: stay idle → cost = gap * power_idle[m]
"""

import numpy as np
from .base import ObjectiveFunction
from ..core.solution import ScheduleSolution


class EnergyObjective(ObjectiveFunction):
    """
    Total energy consumption with optimal turn-off decisions.

    Requires instance.power_on, power_idle, power_reset, break_even_point
    to be set on the HFSPInstance.
    """

    name = "energy"

    def compute(self, solution: ScheduleSolution) -> float:
        instance = solution.instance

        if instance.power_on is None or instance.power_idle is None:
            return 0.0

        power_on = instance.power_on
        power_idle = instance.power_idle
        power_reset = instance.power_reset
        break_even = instance.break_even_point

        total_energy = 0.0

        # Group by machine
        machine_ops = {m: [] for m in range(instance.total_machines)}
        for a in solution.assignments:
            machine_ops[a["machine"]].append(a)

        for m in range(instance.total_machines):
            ops = sorted(machine_ops[m], key=lambda a: a["start"])
            if not ops:
                continue

            # Processing energy
            for a in ops:
                duration = a["end"] - a["start"]
                if power_on is not None:
                    total_energy += duration * power_on[m]

            # Idle / turn-off energy between operations
            for i in range(len(ops) - 1):
                idle_gap = ops[i + 1]["start"] - ops[i]["end"]
                if idle_gap <= 1e-9:
                    continue

                if power_reset is not None and break_even is not None:
                    be = break_even[m] if m < len(break_even) else 0
                    if idle_gap >= be and be > 0 and power_reset is not None:
                        # Turn off strategy: idle for break_even time, then off
                        total_energy += be * power_idle[m]
                        if m < len(power_reset):
                            total_energy += power_reset[m]
                    else:
                        total_energy += idle_gap * power_idle[m]
                else:
                    total_energy += idle_gap * power_idle[m]

        solution.energy = total_energy
        return total_energy
