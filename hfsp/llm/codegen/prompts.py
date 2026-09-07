"""提示词: 让 LLM 生成/改进 CodeSpec(contract + 4 槽代码).

关键设计:
  - 每次给 LLM 的是 **完整 CodeSpec = contract(每槽语义) + slots(代码)**,
    而不是光秃秃的代码 —— 语义让 LLM 的优化先验能用上(治"LLM≈随机")。
  - 轻变异: 只改代码(归属清楚, 改进来自代码)。
  - 重变异: 可同时改 contract + 代码(contract 随精英进化)。
  - LLM 返回 JSON {contract:{...}, slots:{...}}, 由 sandbox 编译校验。
"""

import json
from typing import Dict

from .spec import SLOT_NAMES, DEFAULT_CONTRACT

# ctx 里可用的可信原语(给 LLM 的 API 白皮书)
CTX_API = """You write code that runs inside a trusted skeleton that solves a Hybrid Flow
Shop Scheduling Problem (HFSP). The skeleton does decoding/makespan/timing; YOU
only write the four slots below.

A helper object `ctx` is available to your functions with these read-only methods:
    ctx.num_jobs, ctx.num_stages, ctx.total_machines   (ints)
    ctx.machines_per_stage                             (list[int])
    ctx.pt(job, machine)       -> float  processing time
    ctx.stage_of_machine(m)    -> int
    ctx.makespan_of(perm)      -> float  makespan of a job permutation (expensive, use sparingly)
    ctx.apply_op(name, perm, rng) -> list[int]  apply a registered operator
    ctx.list_ops()             -> list[str]      registered operator names
    ctx.perturb_once(perm, rng) -> list[int]     ONE random validated perturbation
    ctx.perturb_n(perm, n, rng) -> list[int]     n random validated perturbations
    ctx.neh(rng)               -> list[int]  NEH constructive start
    ctx.exp(x), ctx.log(x)     -> float  (math helpers — DO NOT import math)

`rng` is a numpy Generator (rng.integers(a,b), rng.random(), rng.shuffle(list)).

SAFE WRITING RULE (IMPORTANT): in `move`, prefer ORCHESTRATING the validated
ctx helpers above over writing raw index arithmetic. Raw `perm[i:j]=...` /
`rng.integers` off-by-one is the #1 source of rejected code. If you must index,
guard every bound with min/max so it is valid for ANY list length.
"""

_SYSTEM_HEAD = """\
You are an expert at designing metaheuristic neighbourhood operators for the \
Hybrid Flow Shop Scheduling Problem (HFSP, minimizes makespan). A solution is a \
permutation of job ids. A strong operator exploits problem structure to escape \
local optima (block relocation, critical-path aware moves, compound moves), not \
just a trivial single swap.

{CTX_API}
You write FOUR functions that together define an algorithm (the skeleton supplies \
the loop):
    construct(ctx, rng)               -> list[int]   initial job permutation
    move(perm, ctx, rng)              -> list[int]   produce a candidate neighbour
    accept(delta, T, it, ctx, rng)    -> bool        delta=cand_ms-cur_ms (<0 better)
    anneal(T, it, ctx)                -> float       temperature for next iteration

Rules:
1. No imports (no `import math` — use ctx.exp/ctx.log), no I/O, no print.
2. Keep permutations valid (same elements, same length). Never drop/duplicate jobs.
3. **move must NOT modify its input `perm`** — always build and return a NEW list.
   Index carefully: all accesses must stay within [0, len(perm)) on EVERY list size.
4. move must change the permutation and should NOT be a trivial swap if you can
   invent something structurally stronger.
5. `ctx` methods are trusted; use them freely (esp. makespan_of for greedy repair,
   perturb_once / perturb_n / apply_op to compose operators). **Prefer orchestrating
   ctx.perturb_once / ctx.perturb_n / ctx.apply_op over raw index arithmetic** —
   raw bounds bugs are the most common rejection reason.

CORRECT ctx API usage (memorize — most rejections come from these mistakes):
   ctx.perturb_once(perm, rng)            # YES — method on ctx, one arg perm + rng
   ctx.perturb_n(perm, 2, rng)            # YES — n is a plain int
   ctx.apply_op("block", perm, rng)       # YES — name is a STRING from ctx.list_ops()
   ctx.apply_op("perturb_once", ...)      # NO — perturb_once is NOT an operator name
   ctx.makespan_of(new_perm)              # YES — evaluate a permutation (expensive, sparing)
   result = list(perm); result.pop(i)     # YES — plain list ops
   perm.insert(pos, job)                  # NO — mutates input; build `result` and use result.insert
Common operators (ctx.list_ops()): swap, insert, inverse, block, scramble.

6. Output ONLY a single JSON object:
   {{
     "contract": {{ "construct": "<your semantic description>", "move": "...",
                    "accept": "...", "anneal": "..." }},
     "slots": {{ "construct": "def construct(ctx, rng):\\n    ...",
                 "move": "def move(perm, ctx, rng):\\n    ...",
                 "accept": "...", "anneal": "..." }}
   }}
   No markdown, no code fences outside the JSON string values.
"""

# 系统提示(带每槽默认契约当"语义参考", LLM 会改写)
SYSTEM = _SYSTEM_HEAD.format(CTX_API=CTX_API)

# 给 LLM 看当前种子的水平作参照(可选, 提升稳定性)
def _fmt_spec(spec) -> str:
    return json.dumps(spec.to_dict(), indent=2)


def build_mutate_prompt(parent, fitness: float, mode: str,
                        reflection: str = None, pop_best: float = None,
                        insights: str = None,
                        last_error: str = None) -> str:
    """改进一个 parent CodeSpec。

    mode: "light"   -> 只改代码(尤其 move), 保持 contract 与其它槽;
          "medium"  -> 改 1-2 个槽的代码, contract 可微调;
          "heavy"   -> 可以重写 contract + 所有槽(重新设计这个算子).
    """
    lines = []
    if reflection:
        lines.append(f"Analysis of why it may be underperforming: {reflection}\n")
    if insights:
        lines.append(f"Cross-generation insights: {insights}\n")
    lines.append(f"Current candidate (RPD {fitness:+.2f}%):")
    lines.append(_fmt_spec(parent))
    if pop_best is not None:
        lines.append(f"\nCurrent best RPD this generation: {pop_best:+.2f}%")
    lines.append("")
    if mode == "light":
        lines.append("TASK (LIGHT — SMALL SURGERY ONLY): improve the existing code "
                     "by editing AT MOST 2-3 LINES of `move`. Copy the current "
                     "`move` function VERBATIM except for that small edit.")
        lines.append("")
        lines.append("Current `move` code (edit this; keep everything else identical):")
        lines.append("```python")
        lines.append(parent.slots.get("move", "# no move slot"))
        lines.append("```")
        lines.append("")
        lines.append("PREFERRED small edit: switch which validated ctx helper the "
                     "move uses (e.g. ctx.perturb_once vs ctx.perturb_n, a different "
                     "ctx.apply_op operator name, a different number of steps) — "
                     "NOT new raw index code. If you keep raw indexing, guard every "
                     "bound with min/max for any list length. DO NOT rewrite the "
                     "function from scratch. DO NOT change construct / accept / "
                     "anneal. DO NOT change the contract.")
    elif mode == "medium":
        lines.append("TASK (MEDIUM): improve the `move` STRUCTURE — write a "
                     "move that is structurally DIFFERENT from the parent's, not "
                     "a parameter tweak. Ideas: a destructive-reconstructive move "
                     "(remove a random block then re-insert its jobs at best "
                     "positions via ctx.makespan_of), a multi-operator compound, "
                     "or an adaptive-strength perturbation. You may keep or "
                     "rewrite the contract; keep construct / accept / anneal "
                     "mostly unchanged.")
        lines.append("")
        lines.append("Parent `move` (make yours structurally different):")
        lines.append("```python")
        lines.append(parent.slots.get("move", "# no move slot"))
        lines.append("```")
    else:  # heavy
        lines.append("TASK (HEAVY): redesign this operator. You MAY rewrite the "
                     "contract to capture a better design intent, and rewrite any "
                     "slot code. Aim for a structurally different, stronger move. "
                     "Output the full JSON.")
    if last_error:
        lines.append("\nIMPORTANT — your previous attempt was REJECTED for this "
                     "reason; fix it (pay attention to index bounds and the input "
                     "permutation not being modified):")
        lines.append(f"  ERROR: {last_error}")
    lines.append("\nOutput ONLY the JSON object.")
    return "\n".join(lines)


def build_generation_prompt(seed_specs, flavour: str = None) -> str:
    """从零生成一个候选(重变异/初始化用)。给几个种子当参考。"""
    lines = [f"Existing operators (reference only; do not copy verbatim):"]
    for s in seed_specs[:3]:
        lines.append(json.dumps(s.slots.get("move", ""))[:600])
        lines.append("")
    if flavour:
        lines.append(f"\nInvent a NEW move with this flavour: {flavour}\n")
    else:
        lines.append("\nInvent a new, structurally distinct neighbourhood move.\n")
    lines.append("Output ONLY the JSON {contract:{...}, slots:{...}} object.")
    return "\n".join(lines)


def contract_to_text(spec) -> str:
    """把 contract 渲染成可读的"设计意图"文本(落盘/给 LLM 当记忆)."""
    return "\n".join(f"- {k}: {v}" for k, v in spec.contract.items()
                     if k in SLOT_NAMES)
