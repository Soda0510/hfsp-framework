"""
MILP (Mixed Integer Linear Programming) exact solver for HFSP.

Uses gurobipy to formulate and solve the HFSP as a MILP.
Suitable for small instances (n ≤ 20) to verify heuristic solutions.

The model:
    Decision variables:
    - x[j,s,m] ∈ {0,1} : job j at stage s assigned to machine m
    - C[j,s] ≥ 0       : completion time of job j at stage s
    - y[i,j,s,m] ∈ {0,1}: job i precedes job j on machine m at stage s

    Minimize C_max  (makespan)

    Constraints:
    1. Each job assigned to exactly one machine per stage
    2. Completion time ≥ processing time (for stage 0)
    3. Completion time ≥ prev_stage_completion + processing time (stage > 0)
    4. Sequencing: if i before j on machine m, no overlap (big-M)
    5. C_max ≥ completion time at final stage
"""

from typing import Optional
import numpy as np
import time

from ..core.instance import HFSPInstance
from ..core.solution import ScheduleSolution


class MILPSolver:
    """
    MILP exact solver using gurobipy.

    Parameters
    ----------
    time_limit : float
        Solver time limit in seconds.
    mip_gap : float
        Relative MIP optimality gap (e.g., 0.01 = 1%).
    verbose : bool
        If True, print solver output.
    """

    name = "MILP"

    def __init__(
        self,
        time_limit: float = 300.0,
        mip_gap: float = 0.0,
        verbose: bool = False,
    ):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.verbose = verbose

    def solve(self, instance: HFSPInstance) -> ScheduleSolution:
        """
        Solve the HFSP instance exactly via MILP.

        Returns
        -------
        ScheduleSolution
        """
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            raise ImportError(
                "gurobipy is required for MILP solving. "
                "Install with: pip install gurobipy"
            )

        n = instance.num_jobs
        s = instance.num_stages
        pt = instance.processing_times  # (n, total_machines)
        machines_per_stage = instance.machines_per_stage

        # --- Build model ---
        model = gp.Model(f"HFSP_{instance.name}")
        model.setParam("TimeLimit", self.time_limit)
        model.setParam("MIPGap", self.mip_gap)
        if not self.verbose:
            model.setParam("OutputFlag", 0)

        # Big-M (upper bound on completion times)
        big_M = float(np.sum(pt)) * 2

        # Sets
        Jobs = range(n)
        Stages = range(s)

        # Get machine indices per stage
        stage_machines = [list(instance.machines_in_stage(stage)) for stage in Stages]

        # Variables
        # x[j, s, m_local]: job j at stage s assigned to m_local (local index)
        x = {}
        for j in Jobs:
            for st in Stages:
                for m_local, _ in enumerate(stage_machines[st]):
                    x[j, st, m_local] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{st}_{m_local}")

        # Completion times
        C = {}
        for j in Jobs:
            for st in Stages:
                C[j, st] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"C_{j}_{st}")

        # Makespan
        C_max = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="C_max")

        # Sequencing variables: y[i, j, s, m_local] : i before j on machine m at stage s
        y = {}
        for i in Jobs:
            for jj in Jobs:
                if i < jj:
                    for st in Stages:
                        for m_local, _ in enumerate(stage_machines[st]):
                            y[i, jj, st, m_local] = model.addVar(
                                vtype=GRB.BINARY, name=f"y_{i}_{jj}_{st}_{m_local}"
                            )

        # --- Constraints ---

        # 1. Assignment: each job assigned to exactly one machine per stage
        for j in Jobs:
            for st in Stages:
                model.addConstr(
                    gp.quicksum(x[j, st, m_local] for m_local in range(len(stage_machines[st]))) == 1,
                    name=f"assign_{j}_{st}"
                )

        # 2. Completion time constraints
        for j in Jobs:
            for st in Stages:
                n_machines = len(stage_machines[st])
                for m_local in range(n_machines):
                    m_global = stage_machines[st][m_local]
                    p = pt[j, m_global]
                    if st == 0:
                        # First stage: C[j,0] >= sum of processing times on assigned machine
                        model.addConstr(
                            C[j, st] >= p * x[j, st, m_local],
                            name=f"ct_stage0_{j}_{st}_{m_local}"
                        )
                    else:
                        # C[j,s] >= C[j,s-1] + p on assigned machine
                        model.addConstr(
                            C[j, st] >= C[j, st - 1] + p * x[j, st, m_local],
                            name=f"ct_{j}_{st}_{m_local}"
                        )

        # 3. Sequencing: no overlap on same machine
        for i in Jobs:
            for jj in Jobs:
                if i < jj:
                    for st in Stages:
                        for m_local in range(len(stage_machines[st])):
                            m_global = stage_machines[st][m_local]
                            p_jj = pt[jj, m_global]
                            # If i before j on machine m:
                            # C[j,s] >= C[i,s] + p_jj - bigM*(2 - x[i,s,m] - x[j,s,m] - y[i,j,s,m])
                            model.addConstr(
                                C[jj, st] >= C[i, st] + p_jj
                                - big_M * (3 - x[i, st, m_local] - x[jj, st, m_local] - y[i, jj, st, m_local]),
                                name=f"seq1_{i}_{jj}_{st}_{m_local}"
                            )
                            # If j before i on machine m:
                            p_i = pt[i, m_global]
                            model.addConstr(
                                C[i, st] >= C[jj, st] + p_i
                                - big_M * (2 - x[i, st, m_local] - x[jj, st, m_local] + y[i, jj, st, m_local]),
                                name=f"seq2_{i}_{jj}_{st}_{m_local}"
                            )

        # 4. Makespan
        for j in Jobs:
            model.addConstr(C_max >= C[j, s - 1], name=f"makespan_{j}")

        # Objective
        model.setObjective(C_max, GRB.MINIMIZE)

        # --- Solve ---
        model.optimize()

        # --- Extract solution ---
        status = model.Status
        if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            # Build assignments from variable values
            assignments = []
            for j in Jobs:
                for st in Stages:
                    for m_local in range(len(stage_machines[st])):
                        if x[j, st, m_local].X > 0.5:
                            m_global = stage_machines[st][m_local]
                            # Find the start time by checking predecessors
                            start_time = 0.0
                            if st > 0:
                                start_time = max(start_time, C[j, st - 1].X)
                            p = pt[j, m_global]
                            end_time = C[j, st].X

                            assignments.append({
                                "job": j,
                                "stage": st,
                                "machine": m_global,
                                "start": end_time - p,
                                "end": end_time,
                            })

            makespan = C_max.X
            flow_time = sum(C[j, s - 1].X for j in Jobs)

            # Build a dummy permutation (reconstruct from schedule order)
            permutation = self._reconstruct_permutation(instance, assignments)

            solution = ScheduleSolution(
                permutation=permutation,
                instance=instance,
                assignments=assignments,
                makespan=makespan,
                flow_time=flow_time,
                method="MILP",
            )
        else:
            # Infeasible or error — return empty solution
            raise RuntimeError(f"MILP solver failed with status {status}")

        model.dispose()
        return solution

    def _reconstruct_permutation(self, instance, assignments):
        """Reconstruct a job permutation from the schedule (by stage-0 start times)."""
        n = instance.num_jobs
        stage0_start = {}
        for a in assignments:
            if a["stage"] == 0:
                stage0_start[a["job"]] = a["start"]
        # Sort jobs by stage-0 start time
        sorted_jobs = sorted(range(n), key=lambda j: stage0_start.get(j, 0))
        return sorted_jobs
