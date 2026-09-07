"""把进化出的 CodeSpec(骨架+槽)评到 144 实例留出测试集, 对比 P5d 组件基线.

协议与 eval_spec_test.py 一致(rho=2, rng seed 42, 单次运行), 因此可直接与
results/llm/p5d/rpd_table.csv 里的 IG/GA/SA/NEH+LS/... 并列比较。

与 run_code_evolution.py 的 evaluate() 同构: 同一套 SKEL_PARAMS、算子注册表、
decode 预算。区别: 走 test.txt(144) + rho=2 + 单 run(与 P5d 对齐)。

用法:
    .venv/bin/python scripts/eval_codespec_test.py \
        --spec results/llm/code_run/mid5_feedback/best_code.json \
        --label CodeSlot_g3 --rho 2
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hfsp.io import SevilleReader, SevilleReference
from hfsp.llm.codegen.spec import CodeSpec
from hfsp.llm.codegen.sandbox import compile_slots
from hfsp.llm.codegen.ctx import HfspDomainContext
from hfsp.llm.codegen.skeleton import NeutralSkeleton
from hfsp.methods.operators import (SwapOperator, InsertOperator, InverseOperator,
                                    BlockOperator, ScrambleOperator)

SEVILLE = "benchmarks/seville"
TEST_LIST = "benchmarks/seville/split/test.txt"
P5D_CSV = "results/llm/p5d/rpd_table.csv"
OUT_DIR = Path("results/llm/test_evals")

# 与进化 run 一致(骨架超参不随评估变); 大实例预算按 n² 缩放, 不再卡死
SKEL_PARAMS = dict(max_iterations=1000, ls_every=30, ls_iterations=40)
DECODE_BUDGET_MIN = 40000   # 小实例下限; 大实例按 50*n*n 缩放


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="path to best_code.json (CodeSpec)")
    ap.add_argument("--label", required=True, help="e.g. CodeSlot_g3")
    ap.add_argument("--rho", type=float, default=2.0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    d = json.load(open(args.spec))
    spec = CodeSpec(name=d.get("name", args.label), contract=d["contract"],
                    slots=d["slots"], params=d.get("params", {}))
    print(f"evaluating CodeSpec '{args.label}' (name={spec.name})")
    print(f"  move contract: {spec.contract.get('move', '?')[:80]}...")

    op_fns = {k: c().apply for k, c in
              [("swap", SwapOperator), ("insert", InsertOperator),
               ("inverse", InverseOperator), ("block", BlockOperator),
               ("scramble", ScrambleOperator)]}

    reader = SevilleReader(SEVILLE)
    ref = SevilleReference(f"{SEVILLE}/references/UpperBounds_01_April_2019.xlsx")
    names = [ln.strip() for ln in open(TEST_LIST) if ln.strip()]

    # 槽代码与实例无关, 只编译一次; 用中等规模实例做行为校验(贴近真实 n)
    mid_name = sorted(names, key=lambda n: int(n.split("_")[1]))[len(names) // 2]
    _mid_inst = reader.load(mid_name)
    ctx_mid = HfspDomainContext(_mid_inst, operator_fns=op_fns,
                                max_decodes=max(DECODE_BUDGET_MIN,
                                                50 * _mid_inst.num_jobs * _mid_inst.num_jobs))
    ok, fns, msg = compile_slots(spec.slots, ctx_mid)
    if not ok:
        raise SystemExit(f"spec compile failed: {msg}")
    print(f"  compiled slots OK ({msg})\n")

    rows = []
    t0 = time.perf_counter()
    for i, name in enumerate(names):
        inst = reader.load(name)
        bk = ref.best_known(name)
        tl = args.rho * inst.num_jobs * inst.total_machines / 1000.0
        dec_budget = max(DECODE_BUDGET_MIN, 50 * inst.num_jobs * inst.num_jobs)
        ctx = HfspDomainContext(inst, operator_fns=op_fns, max_decodes=dec_budget)
        rng = np.random.default_rng(42)   # 与 P5d 同 seed
        sk = NeutralSkeleton(ctx, fns, rng=rng, time_limit=tl, **SKEL_PARAMS)
        sol = sk.solve(inst)
        rpd = (sol.makespan - bk) / bk * 100.0
        rows.append((args.label, name, rpd))

        if (i + 1) % 15 == 0 or (i + 1) == len(names):
            elapsed = time.perf_counter() - t0
            per = elapsed / (i + 1)
            eta = per * (len(names) - (i + 1))
            print(f"  [{elapsed:>6.0f}s] {i + 1}/{len(names)} instances "
                  f"({per:.1f}s/inst, ETA {eta:.0f}s)")

    out_csv = OUT_DIR / f"{args.label}_rpd.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "instance", "rpd"])
        w.writerows(rows)
    print(f"saved -> {out_csv}")

    # ---- 与 P5d 基线并列 ----
    p5d = pd.read_csv(P5D_CSV)
    mine = pd.read_csv(out_csv)
    merged = pd.concat([p5d, mine], ignore_index=True)
    piv = merged.pivot(index="instance", columns="algorithm", values="rpd")

    print(f"\nmean RPD comparison (144 test set, rho={args.rho}):")
    print(f"{'algorithm':<16}{'mean':>9}{'min':>8}")
    print("-" * 34)
    for a in piv.columns:
        print(f"{a:<16}{piv[a].mean():>9.2f}{piv[a].min():>8.2f}")

    if args.label in piv and "IG" in piv:
        e, o = piv[args.label], piv["IG"]
        w = int((e < o - 1e-9).sum()); l = int((e > o + 1e-9).sum()); t = int((abs(e - o) <= 1e-9).sum())
        print(f"\n{args.label} vs IG: {w}W/{l}L/{t}T (of {len(piv)})")


if __name__ == "__main__":
    main()
