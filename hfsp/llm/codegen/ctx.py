"""DomainContext: 槽代码能碰的"世界" —— 实例只读统计 + 可信算子原语.

设计原则:
  - ctx 是可信代码(不是 LLM 生成的), 所以槽代码可以放心调它。
  - ctx 暴露的内容 = 实例的只读统计 + 一组可复现的排列算子原语。
    不暴露 numpy 本身、不暴露 decoder 内部 —— 避免槽写出不可控/不可复现逻辑。
  - 槽能通过 ctx.makespan_of(perm) 评估候选, 从而写出"贪心重建/局部搜索"
    这类强 move, 而不仅是盲 swap。makespan 评估是可信的(走 decoder)。
  - 将来 VRP: 换一个 ctx 实现即可, 槽代码不变 —— 这是跨问题迁移的接缝。
"""

from typing import List, Optional, Callable, Dict

import os
from pathlib import Path

import numpy as np

from ...core.decoder import ListSchedulingDecoder

# NEH 结果与 spec 无关且昂贵(O(n^2·decode)); 大实例单算可到 25s。
# 两级缓存: 本进程 dict + 磁盘 npy(跨进程/跨 run 共享, 并行 worker 都命中)。
_NEH_CACHE_DIR = Path(__file__).resolve().parents[3] / "results" / "llm" / "neh_cache"
_NEH_CACHE: dict = {}


class HfspDomainContext:
    def __init__(self, instance, decoder: Optional[ListSchedulingDecoder] = None,
                 operator_fns: Optional[Dict[str, Callable]] = None,
                 max_decodes: int = 8000):
        self.instance = instance
        self.decoder = decoder if decoder is not None else ListSchedulingDecoder()
        self._ops = operator_fns or {}
        self._decode_rng = np.random.default_rng(0)   # 稳定 rng: decode 可复现
        self._max_decodes = max_decodes               # 防失控循环的硬上限
        # 每实例的评估计数(诊断 move 内部解码开销用)
        self.decode_calls = 0
        self._neh_cache = None                        # NEH 起点缓存(与 spec 无关)

    # ---- 实例只读统计(全部标量/原生, 不暴露 numpy 数组本体)----
    @property
    def num_jobs(self) -> int:
        return self.instance.num_jobs

    @property
    def num_stages(self) -> int:
        return self.instance.num_stages

    @property
    def total_machines(self) -> int:
        return self.instance.total_machines

    @property
    def machines_per_stage(self) -> List[int]:
        return list(self.instance.machines_per_stage)

    def pt(self, job: int, machine: int) -> float:
        """加工时间 job x global-machine(只读)."""
        return float(self.instance.processing_times[job, machine])

    def stage_of_machine(self, machine: int) -> int:
        return int(self.instance.stage_of_machine(machine))

    # ---- 可信算子原语 ----
    def makespan_of(self, perm: List[int]) -> float:
        """Decode a permutation and return its makespan (trusted)."""
        if self.decode_calls >= self._max_decodes:
            raise RuntimeError("ctx.decode budget exhausted (runaway move?)")
        self.decode_calls += 1
        return float(self.decoder.decode(self.instance, perm, self._decode_rng).makespan)

    def list_ops(self) -> List[str]:
        """Names of the registered (already-validated) operators."""
        return sorted(self._ops)

    def apply_op(self, name: str, perm: List[int], rng) -> List[int]:
        """Apply a registered permutation operator (swap/insert/inverse/...)."""
        if name not in self._ops:
            raise KeyError(f"unknown operator '{name}'; available: {sorted(self._ops)}")
        return self._ops[name](list(perm), rng)

    def perturb_once(self, perm: List[int], rng) -> List[int]:
        """One random validated perturbation (always safe, returns a NEW list).

        This is the SAFE building block for `move`: pick a random operator and
        apply it once. Edge cases are handled inside the operator, so the result
        is guaranteed a valid permutation.
        """
        names = list(self._ops)
        name = names[int(rng.integers(0, len(names)))]
        return self._ops[name](list(perm), rng)

    def perturb_n(self, perm: List[int], n: int, rng) -> List[int]:
        """Apply `n` random validated perturbations (each safe)."""
        out = list(perm)
        for _ in range(max(0, min(int(n), 8))):   # clamp: 0..8 steps
            out = self.perturb_once(out, rng)
        return out

    def neh(self, rng=None) -> List[int]:
        """NEH constructive initial permutation (trusted, cached in dict + disk).

        NEH 是 O(n^2·decode) 且结果与 spec 无关。磁盘缓存让并行 worker /
        多次 run 都命中, 避免每个进程首算一次 25s 的 NEH。
        """
        key = getattr(self.instance, "name", repr(self.instance))
        if key in _NEH_CACHE:
            return list(_NEH_CACHE[key])
        cache_file = _NEH_CACHE_DIR / f"{key}.npy"
        if cache_file.exists():
            arr = np.load(cache_file)
            _NEH_CACHE[key] = [int(x) for x in arr.tolist()]
            return list(_NEH_CACHE[key])
        from ...methods.heuristics import neh_heuristic
        if rng is None:
            rng = self._decode_rng
        perm = list(neh_heuristic(self.instance, self.decoder, rng).permutation)
        _NEH_CACHE[key] = perm
        try:
            _NEH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, np.asarray(perm, dtype=int))
        except OSError:
            pass  # 磁盘缓存失败不致命, 退回内存缓存
        return list(perm)

    # ---- 数学辅助(受控, 免 import math) ----
    def exp(self, x: float) -> float:
        """math.exp — 允许槽做 metropolis 而不 import math."""
        return float(np.exp(x))

    def log(self, x: float) -> float:
        return float(np.log(x)) if x > 0 else float("-inf")
