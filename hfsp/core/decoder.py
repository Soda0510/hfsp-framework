"""
List Scheduling Decoder for HFSP.

Converts a job permutation into a complete schedule using the
"earliest completion time" (ECT) rule at each stage.

Algorithm:
    For each job j in permutation order:
        For each stage s = 0 .. num_stages-1:
            arrival_time = completion time of job j at stage s-1 (or 0 for s=0)
            For each machine m in stage s:
                earliest_start = max(machine_available[m], arrival_time)
                completion = earliest_start + pt(j, m)
            Assign job j to the machine m with the smallest completion time.
            Update machine_available[m] = completion.
"""

from typing import List, Optional
import numpy as np

from .instance import HFSPInstance
from .solution import ScheduleSolution


class ListSchedulingDecoder:
    """
    Decodes a job permutation into a schedule using list scheduling
    with the "first available machine" (ECT) rule.

    Parameters
    ----------
    tie_breaking : str
        "first" (default): pick the first machine with earliest completion.
        "random": pick randomly among tied machines.
    """

    def __init__(self, tie_breaking: str = "first"):
        if tie_breaking not in ("first", "random"):
            raise ValueError(f"tie_breaking must be 'first' or 'random', got '{tie_breaking}'.")
        self.tie_breaking = tie_breaking

    def decode(
        self,
        instance: HFSPInstance,
        permutation: List[int],
        rng: Optional[np.random.Generator] = None,
    ) -> ScheduleSolution:
        """
        Convert a job permutation into a complete schedule.

        Parameters
        ----------
        instance : HFSPInstance
            Problem instance.
        permutation : list[int]
            Job processing order (job indices 0..n-1).
        rng : np.random.Generator, optional
            Random generator for tie-breaking (only needed if tie_breaking="random").

        Returns
        -------
        ScheduleSolution
        """
        n_jobs = instance.num_jobs
        n_stages = instance.num_stages
        m_total = instance.total_machines
        pt = instance.processing_times  # (n_jobs, m_total)

        # Machine available times (when each machine becomes free)
        machine_available = np.zeros(m_total, dtype=float)

        # Job completion times at each stage: job_completion[j, s] = ...
        job_completion = np.zeros((n_jobs, n_stages), dtype=float)

        assignments = []

        # Process jobs in permutation order
        for job in permutation:
            for stage in range(n_stages):
                # Arrival time of this job at this stage
                if stage == 0:
                    arrival = 0.0
                else:
                    arrival = job_completion[job, stage - 1]

                # Get machines in this stage
                machines = instance.machines_in_stage(stage)

                # Compute earliest completion time for each machine in this stage
                best_machine = machines[0]
                best_completion = float("inf")
                tied_machines = []

                for m in machines:
                    start = max(machine_available[m], arrival)
                    p = pt[job, m]
                    completion = start + p

                    if completion < best_completion - 1e-12:
                        best_completion = completion
                        best_machine = m
                        tied_machines = [m]
                    elif abs(completion - best_completion) < 1e-12:
                        tied_machines.append(m)

                # Tie-breaking
                if self.tie_breaking == "random" and len(tied_machines) > 1:
                    if rng is None:
                        rng = np.random.default_rng()
                    best_machine = int(rng.choice(tied_machines))

                start_time = max(machine_available[best_machine], arrival)
                end_time = start_time + pt[job, best_machine]

                # Record assignment
                assignments.append({
                    "job": job,
                    "stage": stage,
                    "machine": best_machine,
                    "start": start_time,
                    "end": end_time,
                })

                # Update state
                machine_available[best_machine] = end_time
                job_completion[job, stage] = end_time

        # Compute makespan
        makespan = float(np.max(job_completion[:, -1])) if n_jobs > 0 else 0.0

        # Compute total flow time
        flow_time = float(np.sum(job_completion[:, -1]))

        solution = ScheduleSolution(
            permutation=list(permutation),
            instance=instance,
            assignments=assignments,
            makespan=makespan,
            flow_time=flow_time,
        )

        return solution
