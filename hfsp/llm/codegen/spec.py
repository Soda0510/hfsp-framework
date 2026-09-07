"""CodeSpec: 骨架+代码槽的算法表示(弃组件菜单后).

一个 CodeSpec = contract + slots + params。

- ``slots``: 每个槽名 -> 一段 Python 代码(由 LLM 生成, 沙箱校验后执行)。
  中立骨架(单解漫步)有 4 个槽:
      construct(ctx, rng)        -> list[int]   初始排列
      move(perm, ctx, rng)       -> list[int]   从当前排列造候选(扰动/破坏重建…)
      accept(delta, T, it, ctx, rng) -> bool    是否接受 (delta = cand_ms - cur_ms)
      anneal(T, it, ctx)         -> float       降温(缺省=几何降温)
- ``contract``: 每个槽名的语义说明(LLM 可改写)。轻变异只改代码; 重变异可
  连同 contract 一起改 —— contract 是 LLM 自己对"怎么写 HFSP 算子"的备忘录,
  随精英存活并沉淀。
- ``params``: 骨架超参(时间预算系数、迭代上限等), 不随代码进化(本次固定)。

类别(SA/IG/…)是槽代码的涌现结果, 不是骨架的前置模板。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List

# 骨架需要的最小槽集合 + 每个槽的默认契约(初版给 LLM 看, 它会改写)
SLOT_NAMES: List[str] = ["construct", "move", "accept", "anneal"]

DEFAULT_CONTRACT: Dict[str, str] = {
    "construct": (
        "Build an initial job permutation (list[int]) for a Hybrid Flow Shop "
        "problem. Prefer a NEH-style or other sensible ordering; you may call "
        "ctx.neh() for a strong NEH start."
    ),
    "move": (
        "Perturb the current permutation into a neighbouring candidate: a "
        "structurally meaningful move that escapes local optima (e.g. remove a "
        "block of jobs and reinsert them, move along a critical region), not a "
        "trivial single swap. Return a NEW list[int], same elements."
    ),
    "accept": (
        "Given delta = candidate_makespan - current_makespan (<0 is better), "
        "temperature T, iteration it, decide whether to accept: True if delta < 0, "
        "or with a temperature-based probability to escape local optima."
    ),
    "anneal": (
        "Return the temperature for the next iteration (decrease over time to "
        "move from exploration to exploitation). Geometric cooling is the default."
    ),
}


@dataclass
class CodeSpec:
    name: str = "candidate"
    contract: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CONTRACT))
    slots: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        return (f"CodeSpec(name={self.name!r}, slots={sorted(self.slots)}, "
                f"params={self.params})")


def validate_spec_structure(spec: CodeSpec) -> List[str]:
    """字段级校验(不执行代码)。返回错误列表, 空 = 结构合法."""
    errors: List[str] = []
    missing = [s for s in SLOT_NAMES if s not in spec.slots]
    if missing:
        errors.append(f"missing slots: {missing}")
    extra = [s for s in spec.slots if s not in SLOT_NAMES]
    if extra:
        errors.append(f"unknown slots: {extra}")
    for s in SLOT_NAMES:
        c = spec.contract.get(s, "")
        if not c or not c.strip():
            errors.append(f"empty contract for slot '{s}'")
    return errors
