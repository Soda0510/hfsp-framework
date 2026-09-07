"""CodeMutator: LLM 改进/生成 CodeSpec(contract + slots).

与组件版 LLMMutator 同接口思路:
  - 提示含 parent 完整 CodeSpec + fitness + contract 语义
  - LLM 返回 JSON {contract, slots} -> sandbox 编译 -> 合法则用, 非法 retry,
    最终回退 parent(mutation no-op)。
区别: 组件版校验是枚举合法性; 这里是 **代码能编译 + 行为合法**(sandbox)。
"""

import json
from typing import Any, Dict, Optional

import numpy as np

from ..client import LLMClient
from .sandbox import compile_slots, _extract_json_object
from .spec import CodeSpec, validate_spec_structure, SLOT_NAMES, DEFAULT_CONTRACT
from .prompts import SYSTEM, build_mutate_prompt, build_generation_prompt
from .ctx import HfspDomainContext


class CodeMutator:
    def __init__(self, client: LLMClient, ctx_builder, max_retries: int = 3,
                 temperature: Optional[float] = None, verbose: bool = True,
                 seed_specs=None):
        """
        ctx_builder: callable(instance) -> HfspDomainContext(用于行为校验).
                    在进化脚本里定义, 绑定算例 + 算子注册表。
        """
        self.client = client
        self.ctx_builder = ctx_builder
        self.max_retries = max_retries
        self.temperature = temperature
        self.verbose = verbose
        self.seed_specs = seed_specs or []
        self.stats = {"calls": 0, "legal": 0, "illegal": 0, "fell_back": 0}

    # ------------------------------------------------------------------
    # 解析 + 校验 LLM 返回
    # ------------------------------------------------------------------

    def _parse(self, raw: dict, ctx) -> tuple:
        """把 LLM 返回 dict -> CodeSpec, 编译槽。返回 (CodeSpec, fns) 或抛错."""
        contract = {k: str(raw.get("contract", {}).get(k, DEFAULT_CONTRACT[k]))
                    for k in SLOT_NAMES}
        slots_raw = raw.get("slots", {})
        # slots 可能是字符串代码, 也可能是嵌套 dict {name: code}
        slots = {}
        for k in SLOT_NAMES:
            v = slots_raw.get(k)
            if isinstance(v, dict):
                v = v.get("code") or v.get(k)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"slot '{k}' missing or not a code string")
            slots[k] = v
        spec = CodeSpec(contract=contract, slots=slots,
                        params={"source": "llm"})
        errs = validate_spec_structure(spec)
        if errs:
            raise ValueError("; ".join(errs))
        ok, fns, msg = compile_slots(spec.slots, ctx)
        if not ok:
            raise ValueError(f"slots invalid: {msg}")
        return spec, fns

    # ------------------------------------------------------------------
    # 变异
    # ------------------------------------------------------------------

    def mutate(self, parent: CodeSpec, ctx, rng=None, name: str = "child",
               feedback: Optional[Dict[str, Any]] = None) -> CodeSpec:
        """改进 parent。回退 = 原样 parent(改名)."""
        self.stats["calls"] += 1
        fitness = feedback.get("fitness") if feedback else None
        mode = feedback.get("mode", "medium") if feedback else "medium"
        reflection = feedback.get("reflection") if feedback else None
        pop_best = feedback.get("pop_best") if feedback else None
        insights = feedback.get("insights") if feedback else None

        last_err = None
        user = build_mutate_prompt(
            parent, fitness if fitness is not None else 0.0, mode,
            reflection=reflection, pop_best=pop_best, insights=insights)
        for attempt in range(self.max_retries):
            try:
                raw = self.client.chat_json(SYSTEM, user, temperature=self.temperature)
                spec, _ = self._parse(raw, ctx)
                spec.name = name
                self.stats["legal"] += 1
                return spec
            except Exception as e:
                last_err = e
                self.stats["illegal"] += 1
                if self.verbose:
                    print(f"    [CodeMutator] attempt {attempt + 1}/{self.max_retries} "
                          f"failed ({type(e).__name__}: {str(e)[:120]})")
                # 下一轮把失败原因喂回 LLM(错误回传是合法率的关键杠杆)
                user = build_mutate_prompt(
                    parent, fitness if fitness is not None else 0.0, mode,
                    reflection=reflection, pop_best=pop_best, insights=insights,
                    last_error=str(e)[:400])
        self.stats["fell_back"] += 1
        if self.verbose:
            print(f"    [CodeMutator] giving up — keeping parent '{parent.name}'")
        spec = CodeSpec(name=name, contract=dict(parent.contract),
                        slots=dict(parent.slots), params=dict(parent.params))
        return spec

    def generate(self, ctx, rng=None, name: str = "fresh",
                 flavour: Optional[str] = None) -> Optional[CodeSpec]:
        """从零生成一个候选(种群初始化/重变异)。失败返回 None."""
        self.stats["calls"] += 1
        user = build_generation_prompt(self.seed_specs, flavour)
        for attempt in range(self.max_retries):
            try:
                raw = self.client.chat_json(SYSTEM, user, temperature=self.temperature)
                spec, _ = self._parse(raw, ctx)
                spec.name = name
                self.stats["legal"] += 1
                return spec
            except Exception as e:
                self.stats["illegal"] += 1
                if self.verbose:
                    print(f"    [CodeMutator generate] attempt {attempt + 1} failed: "
                          f"{str(e)[:120]}")
                user = build_generation_prompt(self.seed_specs, flavour)
                user += (f"\n\nIMPORTANT — your previous attempt was REJECTED: "
                         f"{str(e)[:400]}\nFix the bug and retry.")
        self.stats["fell_back"] += 1
        return None
