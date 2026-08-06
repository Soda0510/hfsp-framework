"""
Reader for HFSP instance files in Excel format.

Reads Excel files with the following expected sheets:
- Parameter: Job, Stage, Machine_stage, Total_machine, (tao, R)
- Pt_on_machine: Processing time per job per machine
- [Process Time]: Base processing times per job per stage
- [Speed&Rate]: Per-machine speed and conversion factor
- [Power&TB]: Per-machine power and break-even data
- [Duedate]: Per-job due dates
- [Q_table, Q_table_1, Q_table_2]: Q-learning tables
"""

import os
import glob
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd

from ..core.instance import HFSPInstance


class InstanceReader:
    """
    Reads HFSP instance from Excel files.

    Usage:
        reader = InstanceReader("Data")
        instance = reader.load("10-5-6")
        instances = reader.load_all()
    """

    def __init__(self, data_dir: str = "Data"):
        self.data_dir = Path(data_dir)

    # ---- Public API ----

    def list_instances(self) -> List[str]:
        """Return list of available instance names (without .xlsx extension)."""
        pattern = str(self.data_dir / "*.xlsx")
        files = glob.glob(pattern)
        names = []
        for f in files:
            basename = os.path.basename(f)
            if basename.startswith("~$"):
                continue
            name = basename.replace(".xlsx", "")
            # Skip non-instance files like "Orthogonal Array"
            if self._is_instance_file(name):
                names.append(name)
        return sorted(names, key=self._sort_key)

    def load(self, name: str) -> HFSPInstance:
        """
        Load a single instance by name (e.g., "10-5-6").
        """
        path = self.data_dir / f"{name}.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"Instance file not found: {path}")
        return self._read_instance(path, name)

    def load_all(self) -> List[HFSPInstance]:
        """Load all available instances."""
        return [self.load(name) for name in self.list_instances()]

    # ---- Internal methods ----

    def _is_instance_file(self, name: str) -> bool:
        """Check if filename looks like an HFSP instance."""
        parts = name.split("-")
        if len(parts) == 3:
            return all(p.isdigit() for p in parts)
        return False

    def _sort_key(self, name: str) -> tuple:
        """Sort instances by (jobs, stages, machines)."""
        parts = name.split("-")
        return tuple(int(p) for p in parts)

    def _read_instance(self, path: Path, name: str) -> HFSPInstance:
        """Read a single Excel file and construct an HFSPInstance."""
        # Read all sheets into a dict of DataFrames (header=None)
        sheets = {}
        xls = pd.ExcelFile(path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet_name, header=None)
            sheets[sheet_name] = df

        # Parse Parameter sheet
        param = self._parse_parameter(sheets.get("Parameter"))

        num_jobs = param["Job"]
        num_stages = param["Stage"]
        m_per_stage = param["Machine_stage"]
        total_machines = param["Total_machine"]
        tao = param.get("tao")
        R = param.get("R")

        # Build machines_per_stage list (uniform)
        machines_per_stage = [m_per_stage] * num_stages

        # Parse Pt_on_machine (main processing time data)
        pt_on_machine = self._parse_matrix(
            sheets.get("Pt_on_machine"),
            expected_rows=num_jobs,
            expected_cols=total_machines,
        )

        # Parse optional sheets
        base_pt = self._parse_matrix(
            sheets.get("Process Time"),
            expected_rows=num_jobs,
            expected_cols=num_stages,
        )

        speed = None
        conv_factor = None
        speed_df = sheets.get("Speed&Rate")
        if speed_df is not None:
            speed = self._parse_vector(speed_df, expected_len=total_machines, col=1)
            conv_factor = self._parse_vector(speed_df, expected_len=total_machines, col=2)

        power_on = None
        power_idle = None
        power_reset = None
        break_even = None
        power_df = sheets.get("Power&TB")
        if power_df is not None:
            power_on = self._parse_vector(power_df, expected_len=total_machines, col=1)
            power_idle = self._parse_vector(power_df, expected_len=total_machines, col=2)
            power_reset = self._parse_vector(power_df, expected_len=total_machines, col=3)
            break_even = self._parse_vector(power_df, expected_len=total_machines, col=4)

        due_dates = None
        due_df = sheets.get("Duedate")
        if due_df is not None:
            due_dates = self._parse_vector(due_df, expected_len=num_jobs, col=1)

        # Parse Q-tables
        q_table = self._parse_square_matrix(sheets.get("Q_table"))
        q_table_1 = self._parse_square_matrix(sheets.get("Q_table_1"))
        q_table_2 = self._parse_square_matrix(sheets.get("Q_table_2"))

        instance = HFSPInstance(
            name=name,
            num_jobs=num_jobs,
            num_stages=num_stages,
            machines_per_stage=machines_per_stage,
            total_machines=total_machines,
            processing_times=pt_on_machine,
            base_processing_times=base_pt,
            speed=speed,
            conversion_factor=conv_factor,
            power_on=power_on,
            power_idle=power_idle,
            power_reset=power_reset,
            break_even_point=break_even,
            due_dates=due_dates,
            tao=tao,
            R=R,
            q_table=q_table,
            q_table_1=q_table_1,
            q_table_2=q_table_2,
        )

        return instance

    # ---- Parsing helpers ----

    @staticmethod
    def _parse_parameter(df: Optional[pd.DataFrame]) -> dict:
        """Parse Parameter sheet into a dict."""
        if df is None:
            return {}
        params = {}
        for _, row in df.iterrows():
            key = row.iloc[0]
            val = row.iloc[1]
            # Convert numpy types to native Python
            if hasattr(val, "item"):
                val = val.item()
            params[str(key)] = val
        return params

    @staticmethod
    def _parse_matrix(
        df: Optional[pd.DataFrame],
        expected_rows: int,
        expected_cols: int,
    ) -> np.ndarray:
        """
        Parse a DataFrame into a (expected_rows × expected_cols) float matrix.
        Skips the first column (labels like J1, J2, ...) and first row (headers).
        """
        if df is None:
            return np.zeros((expected_rows, expected_cols))

        # Remove first row (header) and first column (row labels)
        data = df.iloc[1:, 1:].values.astype(float)

        if data.shape != (expected_rows, expected_cols):
            # Try without dropping first col (some sheets may not have row labels)
            data2 = df.iloc[1:, :].values.astype(float)
            if data2.shape[1] == expected_cols:
                data = data2
            else:
                data3 = df.iloc[:, 1:].values.astype(float)
                if data3.shape == (expected_rows, expected_cols):
                    data = data3

        # If still mismatched, take what we can
        if data.shape[0] >= expected_rows and data.shape[1] >= expected_cols:
            data = data[:expected_rows, :expected_cols]

        return data

    @staticmethod
    def _parse_vector(
        df: Optional[pd.DataFrame],
        expected_len: int,
        col: int = 1,
    ) -> Optional[np.ndarray]:
        """Parse a single column from a DataFrame (skipping header row)."""
        if df is None:
            return None
        if df.shape[0] < 2:
            return None

        vals = df.iloc[1:, col].values.astype(float)
        if len(vals) >= expected_len:
            return vals[:expected_len]
        return vals

    @staticmethod
    def _parse_square_matrix(df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        """Parse a square matrix from a DataFrame (skipping header row and first col)."""
        if df is None:
            return None
        if df.shape[0] < 2:
            return None

        data = df.iloc[1:, 1:].values.astype(float)
        return data
