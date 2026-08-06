"""
Validation utilities for verifying solution correctness.
"""

import numpy as np


def validate_solution(solution: "ScheduleSolution") -> bool:
    """
    Check that a schedule solution is valid:
    1. All jobs are scheduled at all stages.
    2. No machine processes two jobs simultaneously.
    3. Each job passes through stages in order (no time travel).

    Returns True if valid, raises AssertionError otherwise.
    """
    instance = solution.instance
    n_jobs = instance.num_jobs
    n_stages = instance.num_stages

    # Group assignments by machine
    machine_assignments = {m: [] for m in range(instance.total_machines)}
    job_stage_assignments = {(j, s): None for j in range(n_jobs) for s in range(n_stages)}

    for a in solution.assignments:
        machine_assignments[a["machine"]].append(a)
        job_stage_assignments[(a["job"], a["stage"])] = a

    # 1. All job-stage pairs must be scheduled
    for (j, s), a in job_stage_assignments.items():
        assert a is not None, f"Job {j} stage {s} is not scheduled."

    # 2. Check no overlap on each machine
    for m, assignments in machine_assignments.items():
        sorted_ops = sorted(assignments, key=lambda a: a["start"])
        for i in range(len(sorted_ops) - 1):
            assert sorted_ops[i]["end"] <= sorted_ops[i + 1]["start"], (
                f"Machine {m}: overlap between job {sorted_ops[i]['job']} "
                f"(ends {sorted_ops[i]['end']}) and job {sorted_ops[i + 1]['job']} "
                f"(starts {sorted_ops[i + 1]['start']})."
            )

    # 3. Check stage order for each job
    for j in range(n_jobs):
        for s in range(n_stages - 1):
            prev_end = job_stage_assignments[(j, s)]["end"]
            next_start = job_stage_assignments[(j, s + 1)]["start"]
            assert prev_end <= next_start, (
                f"Job {j}: stage {s} ends at {prev_end} but stage {s + 1} "
                f"starts at {next_start}."
            )

    return True
