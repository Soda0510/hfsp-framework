"""重跑 IG(最强手写基线)到 144 测试集, 协议与 CodeSpec 评估完全一致
(同一 seed 42 / 单次运行 / 同一 rho 时间预算), 使对比 apples-to-apples。

与 P5d 的 IG 同配置(max_iterations=50000, use_local_search=True),
唯一区别是 rho —— 这样 IG@rho=10 能直接和 CodeSlot_g3_rho10 对比。

用法:
    .venv/bin/python scripts/eval_ig_test.py --rho 10 --label IG_rho10
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from hfsp.io import SevilleReader, SevilleReference
from hfsp.methods.metaheuristics import IteratedGreedy

SEVILLE = "benchmarks/seville"
TEST_LIST = "benchmarks/seville/split/test.txt"
OUT_DIR = Path("results/llm/test_evals")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--label", type=str, default="IG")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reader = SevilleReader(SEVILLE)
    ref = SevilleReference(f"{SEVILLE}/references/UpperBounds_01_April_2019.xlsx")
    names = [ln.strip() for ln in open(TEST_LIST) if ln.strip()]

    rows = []
    t0 = time.perf_counter()
    for i, name in enumerate(names):
        inst = reader.load(name)
        bk = ref.best_known(name)
        tl = args.rho * inst.num_jobs * inst.total_machines / 1000.0
        rng = np.random.default_rng(42)   # 与 CodeSlot 评估同 seed
        sol = IteratedGreedy(max_iterations=50000, use_local_search=True,
                             rng=rng, time_limit=tl).solve(inst)
        rpd = (sol.makespan - bk) / bk * 100.0
        rows.append((args.label, name, rpd))

        if (i + 1) % 15 == 0 or (i + 1) == len(names):
            elapsed = time.perf_counter() - t0
            per = elapsed / (i + 1)
            eta = per * (len(names) - (i + 1))
            print(f"  [{elapsed:>6.0f}s] {i + 1}/{len(names)} "
                  f"({per:.1f}s/inst, ETA {eta:.0f}s)")

    out_csv = OUT_DIR / f"{args.label}_rpd.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "instance", "rpd"])
        w.writerows(rows)
    print(f"saved -> {out_csv}")
    print(f"IG mean RPD @ rho={args.rho}: {np.mean([r[2] for r in rows]):.3f}%")


if __name__ == "__main__":
    main()
