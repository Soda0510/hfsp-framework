"""
Reader for the Fernández-Viagas & Framinan (2020) HFSP testbed.

Two pieces of infrastructure:
- ``SevilleReader``   : parses ``instancia_<n>_<s>_<idx>.txt`` -> ``HFSPInstance``
- ``SevilleReference``: loads the published per-instance upper bounds
  (``UpperBounds_01_April_2019.xlsx``) used as the RPD denominator.

Instance file format (identical machines):
    line 1:  <num_jobs> <num_stages>
    line 2:  <machines_per_stage>            # one value per stage
    lines 3+: <num_stages> rows x <num_jobs> # processing times, stage-major

Within a stage all machines are identical, so job j has the same processing
time on every machine of stage s: ``pt[j][m] = p[stage(j)][j]``.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..core.instance import HFSPInstance

# <n>_<s>_<idx>
_NAME_RE = re.compile(r"instancia_(\d+)_(\d+)_(\d+)")


def parse_name(name: str) -> Tuple[int, int, int]:
    """Return (n_jobs, n_stages, idx) for an instance name."""
    m = _NAME_RE.search(name)
    if not m:
        raise ValueError(f"Not a valid instancia name: {name!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


class SevilleReader:
    """Reads the 480 `instancia_*.txt` instances."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.small_dir = self.root / "instances" / "small"
        self.big_dir = self.root / "instances" / "big"

    def list_instances(self, size: str = "all") -> List[str]:
        """Return sorted instance names (without .txt); size in {'small','big','all'}."""
        names = []
        if size in ("all", "small"):
            names += [p.stem for p in self.small_dir.glob("instancia_*.txt")]
        if size in ("all", "big"):
            names += [p.stem for p in self.big_dir.glob("instancia_*.txt")]
        return sorted(names, key=lambda n: parse_name(n))

    def load(self, name: str) -> HFSPInstance:
        path = self._find(name)
        n, s, _ = parse_name(name)
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        assert len(lines) >= 2 + s, f"{name}: expected >= {2 + s} lines, got {len(lines)}"

        header_n, header_s = map(int, lines[0].split())
        assert (header_n, header_s) == (n, s), f"{name}: header {header_n},{header_s} != {n},{s}"

        machines = [int(x) for x in lines[1].split()]
        assert len(machines) == s, f"{name}: expected {s} machine counts, got {len(machines)}"

        # Stage-major processing times: s rows x n columns.
        stage_pt = []
        for i in range(s):
            row = [float(x) for x in lines[2 + i].split()]
            assert len(row) == n, f"{name}: stage {i} has {len(row)} values, expected {n}"
            stage_pt.append(row)

        total_m = int(sum(machines))
        pt = np.zeros((n, total_m), dtype=float)
        m_start = 0
        for stage in range(s):
            for _ in range(machines[stage]):
                for j in range(n):
                    pt[j, m_start] = stage_pt[stage][j]
                m_start += 1

        return HFSPInstance(
            name=name,
            num_jobs=n,
            num_stages=s,
            machines_per_stage=machines,
            total_machines=total_m,
            processing_times=pt,
            base_processing_times=np.array(stage_pt, dtype=float).T,  # (n, s) job-major
        )

    def load_all(self, size: str = "all") -> List[HFSPInstance]:
        return [self.load(n) for n in self.list_instances(size)]

    # ------------------------------------------------------------------

    def _find(self, name: str) -> Path:
        for d in (self.small_dir, self.big_dir):
            p = d / f"{name}.txt"
            if p.exists():
                return p
        raise FileNotFoundError(f"Instance {name!r} not found under {self.root}")


class SevilleReference:
    """Published per-instance upper bounds (best-known as of the testbed paper)."""

    def __init__(self, xlsx_path: str):
        xls = pd.ExcelFile(xlsx_path)
        dfs = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            dfs.append(df)
        ref = pd.concat(dfs, ignore_index=True)
        self._data: Dict[Tuple[int, int, int], float] = {
            (int(r["n"]), int(r["s"]), int(r["Instance"])): float(r["Cmax"])
            for _, r in ref.iterrows()
        }

    def best_known(self, name: str) -> float:
        """Return the published upper bound for an instance name."""
        key = parse_name(name)
        return self._data[key]

    def best_known_dict(self) -> Dict[str, float]:
        """All references keyed by instance name."""
        return {
            f"instancia_{n}_{s}_{idx}": cmax
            for (n, s, idx), cmax in self._data.items()
        }

    def __len__(self) -> int:
        return len(self._data)
