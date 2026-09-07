"""
Oracle Gap 诊断脚本(阶段0 · 第一个决定性实验)。

目的:用零 LLM 调用、零训练,分辨"搜得动却搜不出更好"的三个死因:
  死因1 找错目标  —— 一直找单一最优(SBS),但算法各有所长 → oracle gap 大
  死因2 算法池同质 —— 各算法行为雷同 → 交互残差弱(交互/噪声比 < 1.3)
  死因3 改进被噪声淹 —— seed 噪声 > 代际改进 → 灰区,需加 seed

方法:二维方差分解(实例效应 + 算法效应 + 交互残差)+ 胜者诅咒 seed 分组。
判据见《Oracle_Gap诊断方案.txt》:
  交互/噪声 > 2    → 有真实特化(死因1, 转每实例自适应)
  交互/噪声 < 1.3  → 池子同质(死因2, 换主张/重做组件空间)
  1.3 ~ 2          → 灰区(先加 seed 到 8~10 压噪声重测)

用法:
  python3 oracle_gap.py --results results.csv --bks bks.csv --out report

results.csv 列(可缺 generation):
  algo_id, instance_id, seed, makespan[, generation]
  (列名容错: algorithm/instance/run 也认)
bks.csv 列:
  instance_id, bks
"""

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 数据读入 + 列名容错
# ---------------------------------------------------------------------------

_ALIAS = {
    "algo_id": ["algo_id", "algorithm", "algo", "solver"],
    "instance_id": ["instance_id", "instance"],
    "seed": ["seed", "run"],
    "makespan": ["makespan", "obj", "cost"],
    "generation": ["generation", "gen"],
}


def _resolve(cols: List[str], key: str) -> str:
    for cand in _ALIAS[key]:
        if cand in cols:
            return cand
    return key  # fallback to the canonical name; read will error clearly if absent


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)
    ren = {}
    for key in ("algo_id", "instance_id", "seed", "makespan", "generation"):
        src = _resolve(cols, key)
        if src in df.columns:
            ren[src] = key
    df = df.rename(columns=ren)
    for req in ("algo_id", "instance_id", "seed", "makespan"):
        if req not in df.columns:
            raise SystemExit(f"results.csv 缺列: {req}(现有: {list(df.columns)})")
    df["algo_id"] = df["algo_id"].astype(str)
    df["instance_id"] = df["instance_id"].astype(str)
    return df


def load_bks(path: str) -> Dict[str, float]:
    df = pd.read_csv(path)
    icol = "instance_id" if "instance_id" in df.columns else df.columns[0]
    bcol = "bks" if "bks" in df.columns else df.columns[1]
    df[icol] = df[icol].astype(str)
    return dict(zip(df[icol], df[bcol].astype(float)))


# ---------------------------------------------------------------------------
# 核心计算
# ---------------------------------------------------------------------------

def compute_rpd(df: pd.DataFrame, bks: Dict[str, float]) -> pd.DataFrame:
    """(makespan - bks) / bks * 100,逐 (algo, instance, seed)。"""
    d = df.copy()
    d["bks"] = d["instance_id"].map(bks)
    miss = d["bks"].isna()
    if miss.any():
        bad = sorted(set(d.loc[miss, "instance_id"]))
        raise SystemExit(f"bks.csv 缺少这些实例的上界: {bad[:10]}")
    d["rpd"] = (d["makespan"] - d["bks"]) / d["bks"].abs() * 100.0
    return d


def seed_split(seeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """胜者诅咒防护:seeds 对半分,A 组挑算法,B 组报成绩。"""
    s = np.sort(np.unique(seeds))
    k = max(1, len(s) // 2)
    return s[:k], s[k:] if k < len(s) else s[:1]


def variance_decomp(cell_mean: pd.DataFrame) -> Tuple[float, float, float]:
    """二维方差分解,返回 (交互残差方差, 实例效应方差, 算法效应方差)。"""
    M = cell_mean.values  # (algo, instance)
    grand = M.mean()
    algo_eff = M.mean(axis=1) - grand          # 每个算法整体强弱
    inst_eff = M.mean(axis=0) - grand          # 每个实例本来难易
    resid = M - grand - algo_eff[:, None] - inst_eff[None, :]
    return float(np.var(resid)), float(np.var(inst_eff)), float(np.var(algo_eff))


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle gap 诊断")
    ap.add_argument("--results", required=True)
    ap.add_argument("--bks", required=True)
    ap.add_argument("--out", default="report")
    ap.add_argument("--min-seeds", type=int, default=3,
                    help="低于此 seed 数只算 oracle gap,不算噪声归一化比(默认 3)")
    args = ap.parse_args()

    raw = load_results(args.results)
    bks = load_bks(args.bks)
    d = compute_rpd(raw, bks)

    n_seed = d["seed"].nunique()
    n_algo = d["algo_id"].nunique()
    n_inst = d["instance_id"].nunique()
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("Oracle Gap 诊断报告")
    lines.append(f"算法 {n_algo} 个 × 实例 {n_inst} 个 × seed {n_seed} 个")
    lines.append("=" * 60)

    if n_seed < 2:
        lines.append("⚠ 只有 1 个 seed:无法区分噪声与特化,且 oracle gap 会虚高")
        lines.append("  (胜者诅咒)。以下数字仅作第一眼参考,补 seed 后再信结论。")

    # 每个 (algo, instance) 的 seed 均值矩阵
    cell_mean = d.groupby(["algo_id", "instance_id"])["rpd"].mean().unstack()

    # ---- SBS vs VBS(用 seed 分组防胜者诅咒)----
    if n_seed >= 2:
        sA, sB = seed_split(np.sort(d["seed"].unique()))
        pick = d[d["seed"].isin(sA)].groupby(
            ["algo_id", "instance_id"])["rpd"].mean().unstack()   # 挑算法用 A
        rep = d[d["seed"].isin(sB)].groupby(
            ["algo_id", "instance_id"])["rpd"].mean().unstack()   # 报成绩用 B
        # 每个实例:在 A 组上挑最好的算法,用 B 组成绩报
        best_algo_per_inst = pick.idxmin(axis=0)
        vbs = np.mean([rep.loc[best_algo_per_inst[i], i]
                       for i in pick.columns if i in rep.columns])
        # SBS:整体最好的单一算法(在 A 组上挑),B 组报
        sbs_algo = pick.mean(axis=1).idxmin()
        sbs = float(rep.loc[sbs_algo].mean())
    else:
        pick = cell_mean
        best_algo_per_inst = pick.idxmin(axis=0)
        vbs = float(np.nanmean([pick.loc[best_algo_per_inst[i], i]
                                for i in pick.columns]))
        sbs_algo = pick.mean(axis=1).idxmin()
        sbs = float(pick.loc[sbs_algo].mean())

    gap = sbs - vbs                      # 绝对 gap(RPD 百分点)
    rel_gap = gap / sbs * 100 if sbs != 0 else float("nan")

    # ---- 赢家集中度:每个实例谁最好(用挑算法的表)----
    winners = pick.idxmin(axis=0)
    conc = winners.value_counts(normalize=True).iloc[0] * 100

    lines.append("")
    lines.append("[一] 目标对不对(死因1)")
    lines.append(f"  SBS(单一最优 {sbs_algo})  = {sbs:+.2f}%")
    lines.append(f"  VBS/oracle(每实例挑最好)  = {vbs:+.2f}%")
    lines.append(f"  oracle gap 绝对 = {gap:+.2f} 个百分点")
    lines.append(f"  oracle gap 相对 = {rel_gap:+.1f}%")
    lines.append(f"  赢家集中度      = {conc:.0f}% "
                 f"(>50% 说明单个算法霸榜 → 策略塌缩的证据)")

    # ---- 交互/噪声比(死因2, 只在多 seed 下有意义)----
    if n_seed >= 2:
        inter_var, inst_var, algo_var = variance_decomp(cell_mean)
        # 噪声:每个 (algo, instance) 格子里 seed 间的方差,取平均
        noise_var = float(d.groupby(["algo_id", "instance_id"])["rpd"]
                          .var().mean())
        ratio = inter_var / noise_var if noise_var > 0 else float("inf")
        lines.append("")
        lines.append("[二] 池子同不同质(死因2)")
        lines.append(f"  交互残差方差 = {inter_var:.3f}")
        lines.append(f"  实例效应方差 = {inst_var:.3f}")
        lines.append(f"  算法效应方差 = {algo_var:.3f}")
        lines.append(f"  噪声方差(seed间) = {noise_var:.3f}")
        lines.append(f"  交互/噪声比 = {ratio:.2f}")
    else:
        ratio = None
        lines.append("")
        lines.append("[二] 池子同不同质(死因2):需 ≥2 seed 才能算,暂跳过。")

    # ---- 附加信号 ----
    lines.append("")
    lines.append("[三] 附加信号")
    if sbs < 0 or vbs < 0:
        lines.append("  ⚠ oracle PRD 为负 → 要么 BKS 表过旧,要么评估器有 bug,先查这个。")
    if "generation" in d.columns and d["generation"].nunique() > 1:
        curve = d.groupby("generation")["rpd"].mean().sort_index()
        mono = bool(np.all(np.diff(curve.values) <= 1e-9))
        lines.append(f"  进化曲线单调下降: {mono}(不单调 → 精英保留/选择压力有 bug)")
    else:
        lines.append("  (无 generation 列,跳过进化曲线检查)")

    # ---- 结论 ----
    lines.append("")
    lines.append("[结论]")
    if n_seed >= 2 and ratio is not None:
        if ratio > 2 and rel_gap >= 15:
            lines.append("  → 结局一:算法各有所长,问题在目标(该转每实例自适应)。")
        elif ratio < 1.3:
            lines.append("  → 结局二:池子同质,瓶颈在表示层(换主张/重做组件空间)。")
            lines.append("     ⚠ 此结局下不要上 DPO/GRPO——会把塌缩固化进权重。")
        elif ratio < 2:
            lines.append("  → 结局三(灰区):先加 seed 到 8~10 压噪声重测。")
        else:
            lines.append("  → 交互强但 gap 不够大:再细看逐实例表找结构。")
    else:
        lines.append("  → 数据 seed 不足,结论仅作参考;补 seed 后重跑本脚本。")
        if rel_gap >= 15:
            lines.append("     (oracle gap 已 ≥15%,即便单 seed 也提示死因1 值得深挖。)")

    # ---- 输出 ----
    report = "\n".join(lines)
    print(report)
    with open(f"{args.out}.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # 逐实例明细:每实例选对算法能省多少
    per_inst = pd.DataFrame({
        "instance_id": pick.columns,
        "best_algo": [best_algo_per_inst[i] for i in pick.columns],
        "best_rpd": [pick.loc[best_algo_per_inst[i], i] for i in pick.columns],
        "sbs_rpd": [pick.loc[sbs_algo, i] if sbs_algo in pick.index else np.nan
                    for i in pick.columns],
    })
    per_inst["saved"] = per_inst["sbs_rpd"] - per_inst["best_rpd"]
    per_inst = per_inst.sort_values("saved", ascending=False)
    per_inst.to_csv(f"{args.out}_per_instance.csv", index=False)
    print(f"\n逐实例明细 → {args.out}_per_instance.csv(按 '选对算法能省多少' 降序)")


if __name__ == "__main__":
    main()
