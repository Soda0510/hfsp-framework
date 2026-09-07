"""种子 CodeSpec: 把组件算子 code 化成"完整算法"(含全部 4 槽).

每个种子 = {name, contract(默认), slots{construct/move/accept/anneal}}。
它们与 LLM 生成的 CodeSpec **同构**(都是 code), 在同一 NeutralSkeleton 里跑
作为基线 —— 谁 RPD 低 = 谁是这个骨架下更好的 move, 无骨架差异污染。

construct: 都用 ctx.neh()(NEH 强起点)。
accept: 统一 metropolis 风格 (delta<0 或 exp(-delta/T)) —— 骨架中立,
        accept 只是槽, 种子给一个合理默认。
anneal: 几何降温。
move:   分别 = swap / insert / inverse / block / scramble 的代码转写。
        这些 move 只用 rng + 纯 list 操作, 不碰 ctx.makespan_of ——
        与 LLM 生成的最简 move 同量级, 基线不占 ctx 解码预算的便宜。
"""

from .spec import CodeSpec, DEFAULT_CONTRACT

# 统一的 construct / accept / anneal(所有算子种子共享)
_CONSTRUCT = """def construct(ctx, rng):
    return ctx.neh()
"""

_ACCEPT = """def accept(delta, T, it, ctx, rng):
    if delta < 0:
        return True
    if T <= 0:
        return False
    return rng.random() < (delta / T if delta < T else 1.0) * 0.5
"""

_ANNEAL = """def anneal(T, it, ctx):
    return T * 0.98
"""

_MOVES = {
    "swap": """def move(perm, ctx, rng):
    n = len(perm)
    result = list(perm)
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n))
    while i == j:
        j = int(rng.integers(0, n))
    result[i], result[j] = result[j], result[i]
    return result
""",
    "insert": """def move(perm, ctx, rng):
    n = len(perm)
    result = list(perm)
    from_pos = int(rng.integers(0, n))
    job = result.pop(from_pos)
    to_pos = int(rng.integers(0, n - 1))
    result.insert(to_pos, job)
    return result
""",
    "inverse": """def move(perm, ctx, rng):
    n = len(perm)
    result = list(perm)
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n))
    if i > j:
        i, j = j, i
    if i == j:
        if i > 0:
            i -= 1
        elif j < n - 1:
            j += 1
    result[i:j + 1] = reversed(result[i:j + 1])
    return result
""",
    "block": """def move(perm, ctx, rng):
    n = len(perm)
    if n < 3:
        return list(perm)
    max_len = max(2, n // 3)
    b_len = int(rng.integers(1, max_len + 1))
    if b_len >= n:
        return list(perm)
    start = int(rng.integers(0, n - b_len + 1))
    block = perm[start:start + b_len]
    rest = perm[:start] + perm[start + b_len:]
    if len(rest) == 0:
        return list(perm)
    new_pos = int(rng.integers(0, len(rest) + 1))
    guard = 0
    while new_pos == start and guard < 5:
        new_pos = int(rng.integers(0, len(rest) + 1))
        guard += 1
    result = rest[:new_pos] + block + rest[new_pos:]
    return result
""",
    "scramble": """def move(perm, ctx, rng):
    n = len(perm)
    result = list(perm)
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n))
    if i > j:
        i, j = j, i
    if i == j:
        if i > 0:
            i -= 1
        elif j < n - 1:
            j += 1
    subseq = result[i:j + 1]
    rng.shuffle(subseq)
    result[i:j + 1] = subseq
    return result
""",
    # 示范"编排已验证原语"的安全风格(给 LLM 参照, 也是池里的一员)
    "orchestrator": """def move(perm, ctx, rng):
    # Compose validated primitives instead of raw index math (safer).
    n = len(perm)
    if n < 4:
        return ctx.perturb_once(perm, rng)
    steps = 2 if ctx.makespan_of(perm) > ctx.makespan_of(ctx.perturb_once(perm, rng)) else 1
    out = ctx.perturb_n(perm, steps, rng)
    return out
""",
}

_CONTRACT_OVERRIDES = {
    "swap": "One local move: swap two jobs. Good for fine local refinement.",
    "insert": "One local move: take one job out and reinsert elsewhere.",
    "inverse": "One local move: reverse a contiguous block (larger neighbourhood).",
    "block": "One move: relocate a contiguous block of jobs as a unit — helps escape local optima.",
    "scramble": "One move: randomly shuffle a contiguous subsequence.",
    "orchestrator": "Composite move: applies 1-2 validated perturbations, adapting "
                    "strength to local progress. Shows safe primitive-orchestration style.",
}


def seed_code_specs() -> list:
    """返回 5 个算子种子 CodeSpec(每个 move 只差)。"""
    specs = []
    for op_name, move_src in _MOVES.items():
        contract = dict(DEFAULT_CONTRACT)
        contract["move"] = _CONTRACT_OVERRIDES[op_name]
        specs.append(CodeSpec(
            name=f"seed_{op_name}",
            contract=contract,
            slots={
                "construct": _CONSTRUCT,
                "move": move_src,
                "accept": _ACCEPT,
                "anneal": _ANNEAL,
            },
            params={"operator": op_name},
        ))
    return specs
