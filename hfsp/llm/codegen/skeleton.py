"""NeutralSkeleton: 多槽中立单解漫步骨架.

控制流(可信, 固定) —— 没有任何 SA/GA/IG 标签:
    perm0 = construct(ctx, rng)        # 槽: 初始解
    cur    = decode(perm0); best = cur
    T      = 自适应初温(或 params)
    for it in 1..max_iter:
        if 超时: break
        cand_perm = move(cur.perm, ctx, rng)     # 槽: 造候选
        cand      = decode(cand_perm)
        if accept(cand.ms - cur.ms, T, it, ctx, rng):   # 槽: 接受?
            cur = cand
            if cur.ms < best.ms: best = cur
        T = anneal(T, it, ctx)                    # 槽: 降温
    return best

类别(SA-like / IG-like / …)是槽代码的涌现结果。骨架本身不知道也不关心。

时间预算: 每轮迭代前 _check_time; 初版 max_iterations 是主上限(与组件单解
模板对齐), 时间预算作硬上限。注意组件 executor 有超跑 bug(实测 298%),
我们这里每轮检查 + decode 预算双重保险, 不该超 —— 超了就是 bug。
"""

import time
from typing import Callable, Dict, Optional

import numpy as np

from ...core.decoder import ListSchedulingDecoder
from ...core.solution import ScheduleSolution
from .ctx import HfspDomainContext


class NeutralSkeleton:
    def __init__(
        self,
        ctx: HfspDomainContext,
        slots: Dict[str, Callable],
        *,
        max_iterations: int = 500,
        temperature: Optional[float] = None,   # None -> 自适应初温
        cooling_rate: float = 0.98,
        time_limit: float = float("inf"),
        rng: Optional[np.random.Generator] = None,
        ls_every: int = 25,             # 每隔 N 代对 best 做一次 LS 深潜
        ls_iterations: int = 40,        # LS 内部迭代预算
    ):
        self.ctx = ctx
        self.slots = slots
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.time_limit = time_limit
        self.rng = rng if rng is not None else np.random.default_rng()
        self.start_time = 0.0
        self.convergence: list = []
        self.ls_every = ls_every
        self.ls_iterations = ls_iterations
        self._ls_max_jobs = 80   # 超过此 job 数不做周期性 LS(评估成本控制)

    def _check_time(self) -> bool:
        if self.time_limit == float("inf"):
            return False
        return time.perf_counter() - self.start_time >= self.time_limit

    def solve(self, instance) -> ScheduleSolution:
        self.start_time = time.perf_counter()
        self.convergence = []
        self.ctx.instance = instance   # 每个实例一个 ctx(其内 decoder 绑定 instance)
        # 用默认槽, 防止缺槽直接崩
        slots = self.slots

        # ---- 初始解 ----
        perm = list(slots["construct"](self.ctx, self.rng))
        cur = self.ctx.decoder.decode(instance, perm, self.rng)
        best = cur.copy()
        self.convergence.append(best.makespan)

        # ---- 温度(自适应初温, 与组件单解模板一致) ----
        if self.temperature is not None:
            T = self.temperature
        else:
            T = min(100.0, best.makespan * 0.1) if best.makespan > 0 else 100.0

        from ...methods.operators import local_search as _ls

        for it in range(self.max_iterations):
            if self._check_time():
                break

            # ---- move: 造候选(槽) ----
            try:
                cand_perm = slots["move"](list(cur.permutation), self.ctx, self.rng)
                cand = self.ctx.decoder.decode(instance, cand_perm, self.rng)
            except Exception as e:
                # 失控/异常 move: 放弃本代(保持 cur), 不崩整个求解
                break

            delta = cand.makespan - cur.makespan
            try:
                accepted = bool(slots["accept"](float(delta), float(T), it,
                                                self.ctx, self.rng))
            except Exception:
                accepted = False

            if accepted:
                cur = cand
                if cur.makespan < best.makespan - 1e-12:
                    best = cur.copy()

            # ---- LS 深潜(可信): 每隔 ls_every 代, 对 best 做一轮局部搜索 ----
            # 只在实例小时启用 —— 大实例上 LS(无时间参数, 无改进时扫全邻域)
            # 会严重超时, 而进化评估只求排序, 不需要精确最优。
            if (it + 1) % self.ls_every == 0 and not self._check_time() \
                    and instance.num_jobs <= self._ls_max_jobs:
                try:
                    ls_best = _ls(best, self.ctx.decoder,
                                  max_iterations=self.ls_iterations,
                                  strategy="first_improvement", rng=self.rng)
                    if ls_best.makespan < best.makespan - 1e-12:
                        best = ls_best.copy()
                        cur = ls_best.copy()
                except Exception:
                    pass

            # ---- anneal(槽; 缺省几何降温) ----
            try:
                T = float(slots["anneal"](float(T), it, self.ctx))
            except Exception:
                T = T * self.cooling_rate
            if T < 1e-9:
                T = 1e-9

            self.convergence.append(best.makespan)

        best.method = "NeutralSkeleton"
        return best
