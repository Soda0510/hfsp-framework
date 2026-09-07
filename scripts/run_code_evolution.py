"""骨架+代码槽进化(CodeSpec = contract + slots) —— 主 run.

目标(用户确认): 中立骨架 + 多槽, 跑到"能比过组件"为止; 数据落盘备用后训练。
输出 (results/llm/code_run/):
    meta.json           骨架定义 + 槽schema + 实例集 + rho + bks + 运行参数
    artifacts.jsonl     每个 CodeSpec: code_id, source, parent_id, gen,
                        contract, slots(code), params, legal
    evals.csv           {code_id, instance, seed, makespan, rpd, wall_s}  ← 最细粒度
    best_code.json / evolution_curve.csv / seed_baselines.csv

用法:
    .venv/bin/python scripts/run_code_evolution.py --smoke            # 小规模验证
    .venv/bin/python scripts/run_code_evolution.py --gen 5 --pop 8    # 正式
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from hfsp.io import SevilleReader, SevilleReference
from hfsp.llm.client import OllamaClient
from hfsp.llm.config import LLMConfig
from hfsp.llm.codegen.spec import CodeSpec, SLOT_NAMES
from hfsp.llm.codegen.sandbox import compile_slots
from hfsp.llm.codegen.ctx import HfspDomainContext
from hfsp.llm.codegen.skeleton import NeutralSkeleton
from hfsp.llm.codegen.seeds_code import seed_code_specs
from hfsp.llm.codegen.mutator import CodeMutator
from hfsp.llm.codegen.prompts import contract_to_text

ROOT = "benchmarks/seville"
REF = "benchmarks/seville/references/UpperBounds_01_April_2019.xlsx"
SPLIT_SUBSET = "benchmarks/seville/split/evolution_train_subset.txt"

# 骨架固定超参(与组件对比公平, 不随进化变 —— 只让槽竞争)
SKEL_PARAMS = dict(max_iterations=300, ls_every=30, ls_iterations=40)


def code_id(spec: CodeSpec) -> str:
    d = spec.to_dict()
    d.pop("name", None)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def make_ctx_builder(op_fns, decode_budget: int = 40000):
    def build(inst):
        return HfspDomainContext(inst, operator_fns=op_fns,
                                 max_decodes=decode_budget)
    return build


# ---- 并行评估: fork-global worker(evolve.py 同款模式)----
_WORKER_EVAL = None   # dict with evaluate kwargs


def _worker_fitness_only(spec: CodeSpec):
    """Return (mean_rpd, per_cell) 供 pool.map; per_cell 用于逐实例反思与落盘。"""
    assert _WORKER_EVAL is not None
    kw = _WORKER_EVAL
    fit, cells = evaluate(spec, kw["instances"], kw["refs"], n_runs=kw["n_runs"],
                          time_limit=kw["time_limit"], ctx_builder=kw["ctx_builder"])
    return fit, cells


def evaluate(spec: CodeSpec, instances, refs, rng_base: int = 0, n_runs: int = 1,
             time_limit: float = float("inf"), ctx_builder=None, op_fns=None,
             wall_log=None):
    """Fitness of a CodeSpec: mean RPD over instances × runs (固定 seed).

    wall_log: dict instance->list[(wall_s, decode_calls)] if provided.
    Returns (mean_rpd, per_cell)."""
    per_cell = []
    for i, inst in enumerate(instances):
        ctx = ctx_builder(inst)
        ok, fns, msg = compile_slots(spec.slots, ctx)
        if not ok:
            return float("inf"), None   # 应已被外层校验, 防御
        ms_runs = []
        for run in range(n_runs):
            rng = np.random.default_rng(rng_base + i * 1000 + run)
            sk = NeutralSkeleton(ctx, fns, rng=rng, time_limit=time_limit,
                                 **SKEL_PARAMS)
            sol = sk.solve(inst)
            ms_runs.append(sol.makespan)
        mean_ms = float(np.mean(ms_runs))
        rpd = (mean_ms - refs[i]) / refs[i] * 100.0
        per_cell.append((inst.name, mean_ms, rpd))
        if wall_log is not None:
            wall_log.setdefault(spec.name if hasattr(spec, 'name') else '', []).append(
                ctx.decode_calls)
    return float(np.mean([c[2] for c in per_cell])), per_cell


def _parallel_eval(specs, instances, refs, time_limit, ctx_builder, n_runs, jobs):
    """并行评估一批 spec, 返回 {id(spec): (fit, cells_or_None)}."""
    import multiprocessing as mp
    global _WORKER_EVAL
    _WORKER_EVAL = {"instances": instances, "refs": refs,
                    "time_limit": time_limit, "ctx_builder": ctx_builder,
                    "n_runs": n_runs}
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        results = pool.map(_worker_fitness_only, specs)
    _WORKER_EVAL = None
    out = {}
    for s, (fit, cells) in zip(specs, results):
        out[id(s)] = (fit, cells)
    return out


# ---- 行为反馈: 免费启发式反思 + 跨代记忆(治"盲变", 让 LLM 学到行为)----

def build_reflection(spec: CodeSpec, cells):
    """从逐实例 RPD + contract 自述拼一段反思, 告诉 LLM 它弱在哪(零 LLM 成本)."""
    if not cells:
        return None
    mean = float(np.mean([c[2] for c in cells]))
    order = sorted(cells, key=lambda c: c[2], reverse=True)
    worst, best = order[:3], list(reversed(order[-3:]))

    def jobs(n):
        return int(n.split("_")[1])

    def fmt(cs):
        return ", ".join(f"{jobs(n)}-job@{r:.1f}%" for n, _, r in cs)

    lines = [f"Your candidate's overall RPD is {mean:+.2f}%."]
    lines.append(f"Weakest instances (highest RPD): {fmt(worst)}.")
    lines.append(f"Strongest instances (lowest RPD): {fmt(best)}.")
    if min(jobs(n) for n, _, _ in worst) > max(jobs(n) for n, _, _ in best):
        lines.append("PATTERN: fine on SMALL instances but weak on LARGE ones — "
                     "your move likely does not scale (too local / single-job). "
                     "Consider a block-level or more disruptive move on large n.")
    else:
        lines.append("PATTERN: no clear size effect — the weakness is move quality "
                     "itself, not scaling.")
    lines.append(f"Your own contract for `move` says: {spec.contract.get('move', '?')}")
    lines.append("Fix the specific weakness above; do not blind-rewrite.")
    return "\n".join(lines)


def build_insights(best_spec, best_fit, prev_best, diversity, gen):
    """跨代记忆: 告诉下一代 '什么赢过 / 是不是卡住了'."""
    improved = best_fit < prev_best - 1e-9
    lines = [
        f"Generation {gen}: best RPD {best_fit:+.3f}% "
        f"({'IMPROVED' if improved else 'NO improvement'} vs {prev_best:+.3f}%).",
        f"Best move family so far: \"{best_spec.contract.get('move', '?')}\".",
        f"Population diversity: {diversity:.2f}.",
    ]
    if improved:
        lines.append("BUILD ON this winning move family — do not regress to a "
                     "trivial single swap.")
    else:
        lines.append("Nothing improved — population may be stuck; try a "
                     "STRUCTURALLY different move, not a parameter tweak.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=5)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--n-runs", type=int, default=2)
    ap.add_argument("--out", type=str, default="results/llm/code_run")
    ap.add_argument("--smoke", action="store_true",
                    help="小规模验证: 用 6 个 10-job 实例, gen 2")
    ap.add_argument("--no-llm", action="store_true",
                    help="只跑种子基线, 不调 LLM(测 fitness 链)")
    ap.add_argument("--model", type=str, default=None,
                    help="Ollama model tag(默认 qwen2.5:14b)")
    ap.add_argument("--max-instances", type=int, default=None,
                    help="非 smoke 时截断实例数(跨尺寸取样前 N 个)")
    ap.add_argument("--max-jobs", type=int, default=None,
                    help="进化评估的实例 job 数上限(去掉超大实例, NEH 太贵)")
    args = ap.parse_args()

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    reader = SevilleReader(ROOT)
    ref_reader = SevilleReference(REF)

    # ---- 实例集 ----
    if args.smoke:
        names = ["instancia_10_5_1", "instancia_10_10_1", "instancia_10_20_1",
                 "instancia_15_5_1", "instancia_15_10_1", "instancia_15_20_1"]
        args.gen = min(args.gen, 2)
    else:
        names = [ln.strip() for ln in open(SPLIT_SUBSET) if ln.strip()]
        if args.max_jobs:
            names = [n for n in names if int(n.split("_")[1]) <= args.max_jobs]
        if args.max_instances and len(names) > args.max_instances:
            # 跨尺寸取样: 每个 (jobs) 桶内轮询, 保证覆盖大/中/小
            by_jobs = {}
            for nm in names:
                j = int(nm.split("_")[1])
                by_jobs.setdefault(j, []).append(nm)
            picked, round_ = [], 0
            buckets = sorted(by_jobs)
            while len(picked) < args.max_instances:
                for j in buckets:
                    if round_ < len(by_jobs[j]) and len(picked) < args.max_instances:
                        picked.append(by_jobs[j][round_])
                round_ += 1
            names = picked
    instances = [reader.load(n) for n in names]
    refs = [ref_reader.best_known(n) for n in names]
    budget = lambda inst: args.rho * inst.num_jobs * inst.total_machines / 1000.0
    time_limit = max(budget(i) for i in instances)
    print(f"instances: {len(names)} | rho={args.rho} | time_limit={time_limit:.1f}s | "
          f"gen={args.gen} pop={args.pop}")

    # ---- 算子注册表供 ctx ----
    from hfsp.methods.operators import (SwapOperator, InsertOperator, InverseOperator,
                                        BlockOperator, ScrambleOperator)
    op_fns = {k: c().apply for k, c in
              [("swap", SwapOperator), ("insert", InsertOperator),
               ("inverse", InverseOperator), ("block", BlockOperator),
               ("scramble", ScrambleOperator)]}
    ctx_builder = make_ctx_builder(op_fns)

    seeds = seed_code_specs()
    # ---- 种子基线(fitness + 落盘, 并行)----
    seed_rows = []
    if args.jobs > 1 and len(seeds) > 1:
        res = _parallel_eval(seeds, instances, refs, time_limit, ctx_builder,
                             args.n_runs, args.jobs)
        for s in seeds:
            fit, _ = res[id(s)]
            seed_rows.append((s.name, fit))
    else:
        for s in seeds:
            fit, _ = evaluate(s, instances, refs, time_limit=time_limit,
                              ctx_builder=ctx_builder, n_runs=args.n_runs)
            seed_rows.append((s.name, fit))
    for n_, v in seed_rows:
        print(f"  seed {n_:14s} RPD {v:+.3f}%")
    with open(OUT / "seed_baselines.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seed", "rpd"])
        for n_, v in seed_rows:
            w.writerow([n_, f"{v:.4f}"])

    # ---- meta ----
    meta = {
        "skeleton": {"type": "NeutralSkeleton single-walk", "slots": SLOT_NAMES,
                     "params": SKEL_PARAMS,
                     "ls": "trusted periodic local search on best"},
        "instances": names, "rho": args.rho, "time_limit_s": time_limit,
        "n_runs": args.n_runs, "n_seeds_pop": len(seeds),
        "bks_source": "UpperBounds_01_April_2019", "note": "fine-grained evals.csv for post-train",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    if args.no_llm:
        print("[--no-llm] 只跑种子基线, 结束")
        return

    # ---- 进化 ----
    client = OllamaClient(LLMConfig(model=args.model)) if args.model else OllamaClient()
    mutator = CodeMutator(client, ctx_builder, seed_specs=seeds, verbose=True,
                          max_retries=2)
    # 行为校验用中等大小实例(接近真实部署的 n, 别用最小实例当校验器)
    check_inst = sorted(instances, key=lambda i: i.num_jobs)[len(instances) // 2]

    # 种群: 种子 + 变异种子填充到 pop(变异比从零生成成功率高得多;
    # 从零 generate 的成功率低, 不适合填种群 —— 留给重变异/archive)
    population = list(seeds)
    seed_iter = 0
    while len(population) < args.pop:
        parent = seeds[seed_iter % len(seeds)]
        seed_iter += 1
        child = mutator.mutate(parent, ctx_builder(check_inst), name="init_child",
                               feedback={"fitness": 0.0, "mode": "light"})
        if child is not None and code_id(child) != code_id(parent):
            population.append(child)
        else:
            break  # 变异也失败则不强行凑数

    def fitness_of(spec):
        """返回 mean RPD; 非法 spec 返回 +inf(不落盘)."""
        fit, cells = evaluate(spec, instances, refs, time_limit=time_limit,
                              ctx_builder=ctx_builder, n_runs=args.n_runs)
        return fit, cells

    history_best, history_mean, history_diversity = [], [], []
    artifacts = []
    evals_rows = []
    _evals_logged = set()
    cell_cache = {}   # code_id -> per_cell, 供反思/落盘复用

    def log_eval(spec, fit, cells, gen):
        cid = code_id(spec)
        if cid in _evals_logged or cells is None:
            return
        _evals_logged.add(cid)
        for (iname, ms, rpd) in cells:
            evals_rows.append([cid, iname, 0, f"{ms:.1f}", f"{rpd:.4f}", ""])

    t_start = time.perf_counter()
    best_spec, best_fit = None, float("inf")
    fit_cache = {}
    for gen in range(args.gen + 1):
        fits = {}
        # 收集本代未缓存 spec, 批量评估(并行或串行)
        uncached = [s for s in population if code_id(s) not in fit_cache]
        if uncached:
            if args.jobs > 1:
                res = _parallel_eval(uncached, instances, refs, time_limit,
                                     ctx_builder, args.n_runs, args.jobs)
                for s in uncached:
                    fit, cells = res[id(s)]
                    fit_cache[code_id(s)] = fit
                    cell_cache[code_id(s)] = cells
                    log_eval(s, fit, cells, gen)
            else:
                for s in uncached:
                    fit, cells = fitness_of(s)
                    fit_cache[code_id(s)] = fit
                    cell_cache[code_id(s)] = cells
                    log_eval(s, fit, cells, gen)
        for s in population:
            fits[id(s)] = fit_cache[code_id(s)]
        pop_sorted = sorted(population, key=lambda s: fits[id(s)])
        if fits[id(pop_sorted[0])] < best_fit:
            best_fit, best_spec = fits[id(pop_sorted[0])], pop_sorted[0]
        history_best.append(best_fit)
        history_mean.append(float(np.mean([fits[id(s)] for s in population])))
        uniq = len({code_id(s) for s in population})
        history_diversity.append(uniq / max(len(population), 1))
        el = time.perf_counter() - t_start
        print(f"  gen {gen:>2}/{args.gen}  best RPD {best_fit:+.3f}% ({best_spec.name}) "
              f"[{el/60:.1f}min, ETA {(el/(gen+1))*(args.gen-gen)/60:.0f}min]")

        # 落 artifact
        for s in population:
            artifacts.append({
                "code_id": code_id(s), "source": "seed" if s in seeds else "llm",
                "gen": gen, "name": s.name,
                "contract": s.contract, "slots": s.slots, "params": s.params,
            })

        if gen == args.gen:
            break
        # ---- 繁殖: 精英 + 变异(带行为反馈)----
        # 诊断(mid2): light"只改参"无增益 → 大胆结构变异, 选择把关。
        # 60% medium(结构可不同) + 30% light + 10% heavy。
        # 父代池 = top3 精英 + 1 个随机(防近亲)。
        elites = pop_sorted[:2]
        new_pop = list(elites)
        parent_pool = pop_sorted[:max(3, args.pop // 2)] + [pop_sorted[-1]]
        rng_gen = np.random.default_rng(args.seed + gen)
        diversity = history_diversity[-1]
        prev_best = history_best[-2] if len(history_best) > 1 else best_fit
        insights = build_insights(best_spec, best_fit, prev_best, diversity, gen)
        fallback_streak = 0
        while len(new_pop) < args.pop:
            roll = rng_gen.random()
            mode = "medium" if roll < 0.6 else ("light" if roll < 0.9 else "heavy")
            parent = parent_pool[int(rng_gen.integers(0, len(parent_pool)))]
            pf = fits[id(parent)]
            reflection = build_reflection(parent, cell_cache.get(code_id(parent)))
            ctx0 = ctx_builder(check_inst)
            child = mutator.mutate(parent, ctx0, name=f"g{gen+1}",
                                   feedback={"fitness": pf, "mode": mode,
                                             "pop_best": best_fit,
                                             "reflection": reflection,
                                             "insights": insights})
            if code_id(child) == code_id(parent):
                # 变异回退(no-op): 不占槽, 换父重试; 连续失败则接受占槽防死循环
                fallback_streak += 1
                if fallback_streak >= 3:
                    new_pop.append(child)
                    fallback_streak = 0
                continue
            fallback_streak = 0
            new_pop.append(child)
        population = new_pop

    # ---- 落盘 ----
    # 并行模式下 best 的逐实例 cells 可能还没落盘(并行只返 fit), 这里补全:
    if best_spec is not None:
        log_eval(best_spec, best_fit, fitness_of(best_spec)[1], args.gen)
    (OUT / "best_code.json").write_text(json.dumps(best_spec.to_dict(), indent=2))
    with open(OUT / "convergence.json", "w") as f:
        json.dump({"history": history_best, "history_mean": history_mean,
                   "history_diversity": history_diversity}, f, indent=2)
    with open(OUT / "evolution_curve.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["generation", "best_rpd", "mean_rpd", "diversity"])
        for i, (b, m, d) in enumerate(zip(history_best, history_mean, history_diversity)):
            w.writerow([i, f"{b:.4f}", f"{m:.4f}", f"{d:.3f}"])
    with open(OUT / "artifacts.jsonl", "w") as f:
        for a in artifacts:
            f.write(json.dumps(a) + "\n")
    with open(OUT / "evals.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["code_id", "instance", "seed", "makespan", "rpd", "wall_s"])
        for r in evals_rows:
            w.writerow(r)

    print(f"\n=== DONE ===")
    print(f"best: {best_spec.name} RPD {best_fit:+.3f}%  (seed best: "
          f"{min(v for _, v in seed_rows):+.3f}%)")
    print(f"curve: {[f'{h:.2f}' for h in history_best]}")
    print(f"mutator stats: {mutator.stats}")
    print(f"saved -> {OUT}/")


if __name__ == "__main__":
    main()
