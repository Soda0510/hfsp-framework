"""
Statistics computation for experiment results.

RPD (Relative Percentage Deviation), summary statistics (mean, std, min, max).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RunResult:
    """Result of a single algorithm run."""

    instance_name: str
    algorithm_name: str
    run_id: int
    makespan: float
    flow_time: float = 0.0
    tardiness: float = 0.0
    energy: float = 0.0
    weighted_obj: float = 0.0
    runtime: float = 0.0
    convergence: Optional[List[float]] = None
    pareto_front: Optional[List[dict]] = None

    def to_dict(self) -> dict:
        d = {
            "instance": self.instance_name,
            "algorithm": self.algorithm_name,
            "run": self.run_id,
            "makespan": self.makespan,
            "flow_time": self.flow_time,
            "tardiness": self.tardiness,
            "energy": self.energy,
            "weighted_obj": self.weighted_obj,
            "runtime": self.runtime,
        }
        return d


@dataclass
class ExperimentResultSet:
    """Collection of results from a batch experiment."""

    results: List[RunResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results])

    def compute_summary(self) -> pd.DataFrame:
        """
        Group by instance + algorithm, compute summary statistics.
        """
        df = self.to_dataframe()
        summary = df.groupby(["instance", "algorithm"]).agg(
            mean_makespan=("makespan", "mean"),
            std_makespan=("makespan", "std"),
            min_makespan=("makespan", "min"),
            max_makespan=("makespan", "max"),
            mean_runtime=("runtime", "mean"),
            num_runs=("run", "count"),
        ).reset_index()
        return summary

    def compute_rpd(self, best_known: dict = None) -> pd.DataFrame:
        """
        Compute RPD for each result.

        RPD = 100 * (obtained - best_known) / best_known

        If best_known is None, use the minimum makespan per instance as reference.
        """
        df = self.to_dataframe()

        if best_known is None:
            # Use per-instance minimum as reference
            best_known = df.groupby("instance")["makespan"].min().to_dict()

        df["best_known"] = df["instance"].map(best_known)
        df["rpd"] = 100.0 * (df["makespan"] - df["best_known"]) / df["best_known"]
        return df

    def save_csv(self, path: str):
        """Save results as CSV."""
        self.to_dataframe().to_csv(path, index=False)


def compute_rpd(obtained: float, best_known: float) -> float:
    """Compute Relative Percentage Deviation."""
    if best_known == 0:
        return 0.0
    return 100.0 * (obtained - best_known) / best_known
