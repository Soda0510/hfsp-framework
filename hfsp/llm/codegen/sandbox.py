"""Sandbox + 合法性校验 for LLM-generated code slots (多槽).

中立骨架有 4 个槽, 每个槽是一段独立代码(LLM 生成), 用受限 exec 编译:
  - construct(ctx, rng)                -> list[int]     初始排列
  - move(perm, ctx, rng)               -> list[int]     候选排列
  - accept(delta, T, it, ctx, rng)     -> bool          接受?
  - anneal(T, it, ctx)                 -> float         下一轮温度

每个槽用 AST 白名单 + 行为校验(黑盒)。返回可调用函数字典。
ctx 是实例只读 + 可信原语的容器(见 ctx.py); 代码不许 import / IO / 硬编码
具体实例数字 —— 这保证同一段代码可跨实例/跨问题(VRP 后置)复用。
"""

import ast
import builtins
import re
from typing import Callable, Dict, Optional, Tuple

# 槽名 -> (函数签名, 一句话)
SLOT_SIGNATURES = {
    "construct": "def construct(ctx, rng):\n    # return an initial job permutation (list[int])",
    "move": "def move(perm, ctx, rng):\n    # return a NEW candidate permutation (list[int])",
    "accept": "def accept(delta, T, it, ctx, rng):\n    # return bool: accept the move?",
    "anneal": "def anneal(T, it, ctx):\n    # return the temperature for the next iteration (float)",
}

# ---------------------------------------------------------------------------
# 1. AST 白名单
# ---------------------------------------------------------------------------

_ALLOWED_BUILTINS: set = {
    "list", "range", "len", "min", "max", "int", "float", "abs", "sum",
    "enumerate", "reversed", "sorted", "slice", "all", "any", "zip", "round",
    "set", "dict", "tuple", "bool",
}

_ALLOWED_NODES: tuple = (
    ast.Module,
    ast.FunctionDef, ast.Expr, ast.Assign, ast.AugAssign, ast.Return,
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    # py3.14 把若干节点改了名, 新旧都放行(用 getattr 防低版本无新名)
    ast.Delete, getattr(ast, "Del", ast.Delete),
    ast.IfExp, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp,
    ast.List, ast.Tuple, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Attribute, ast.Subscript, ast.Slice, ast.Call, ast.arguments, ast.arg,
    ast.keyword, ast.ListComp, ast.comprehension, ast.Starred, ast.NamedExpr,
    ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.And, ast.Or, ast.Not, ast.USub,
    ast.Div, ast.Pow,
    # 比较运算新名(py3.14): In / NotIn / Is / IsNot
    getattr(ast, "In", ast.In), getattr(ast, "NotIn", ast.NotIn),
    getattr(ast, "Is", ast.Is), getattr(ast, "IsNot", ast.IsNot),
    # 位运算/其它常见
    getattr(ast, "LShift", None), getattr(ast, "RShift", None),
    getattr(ast, "BitAnd", None), getattr(ast, "BitOr", None),
    getattr(ast, "BitXor", None),
)

_FORBIDDEN_NAMES: set = {
    "open", "eval", "exec", "compile", "globals", "locals", "vars",
    "__import__", "setattr", "delattr", "input", "print",
    "exit", "quit", "breakpoint", "type", "object", "isinstance", "issubclass",
    "super", "staticmethod", "classmethod", "property",
    "dir", "help", "memoryview", "iter", "next", "map", "filter", "bytes",
    "str", "repr", "format", "hash", "id", "callable", "complex", "frozenset",
    "Exception", "BaseException", "np", "getattr",
    # 任何以 __ 开头的属性名单独拦截(见 Attribute 检查), 这里放常见逃逸名
    "__class__", "__bases__", "__subclasses__", "__mro__", "__dict__",
    "__globals__", "__code__", "__builtins__",
}


def static_check(src: str) -> Tuple[bool, str]:
    """AST whitelist check for one slot. Returns (ok, reason)."""
    if not src or not src.strip():
        return False, "empty source"
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    # 白名单里可能有 getattr fallback 产生的 None, 先滤掉
    allowed = [n for n in _ALLOWED_NODES if n is not None]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, f"imports are forbidden (line {getattr(node, 'lineno', '?')})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # 构造器/函数调用: 必须在安全 builtin 里且不在黑名单
            if node.func.id in _FORBIDDEN_NAMES or node.func.id not in _ALLOWED_BUILTINS:
                return False, f"forbidden function call: {node.func.id}"
        if isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_NAMES or node.attr.startswith("__"):
                return False, f"forbidden attribute: {node.attr}"
        if isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                return False, f"forbidden name: {node.id}"
        if not isinstance(node, tuple(allowed)):
            return False, f"disallowed syntax: {type(node).__name__}"
    return True, "ok"


# ---------------------------------------------------------------------------
# 2. 行为校验(黑盒, 用 ctx 实例做真实输入)
# ---------------------------------------------------------------------------

def _is_permutation(p, n: int) -> bool:
    return (isinstance(p, list) and len(p) == n and sorted(p) == list(range(n)))


def _behavioral_check(fn: Callable, slot: str, ctx, n_trials: int = 6
                      ) -> Tuple[bool, str]:
    """Run a slot on random permutations through a real ctx; verify output type."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = ctx.num_jobs
    # move 按"真实 HFSP 长度"测: 全量 n 和一个约半长 —— 不用 3/5/7 这类
    # 现实不存在的极小长度(LLM 按真实 HFSP 写边界, 小长度会误杀 low>=high)。
    test_lens = sorted({n, max(6, n // 2)})   # 真实 HFSP 长度; 不测 n//4(太极端)
    try:
        if slot == "construct":
            perm = fn(ctx, rng)
            if not _is_permutation(perm, n):
                return False, f"construct returned invalid perm: {perm!r}"
        elif slot == "move":
            for L in test_lens:
                for _ in range(n_trials):
                    base = list(rng.permutation(L))
                    base_copy = list(base)
                    out = fn(list(base_copy), ctx, rng)
                    if not _is_permutation(out, L):
                        return False, f"move returned invalid perm at len {L}: {out!r}"
                    # move 不得原地改输入(契约: 返回新 list)
                    if base_copy != base:
                        return False, f"move MUTATED its input permutation at len {L}"
        elif slot == "accept":
            for _ in range(n_trials):
                v = fn(float(rng.normal()), float(rng.uniform(0.1, 50)), 0, ctx, rng)
                if not isinstance(v, (bool, np.bool_)) and not (isinstance(v, int) and v in (0, 1)):
                    return False, f"accept returned {v!r}, expected bool"
        elif slot == "anneal":
            v = fn(100.0, 0, ctx)
            if not isinstance(v, (float, int)):
                return False, f"anneal returned {v!r}, expected number"
    except Exception as e:
        return False, f"raised at runtime: {type(e).__name__}: {e}"
    return True, "ok"


# ---------------------------------------------------------------------------
# 3. 隔离执行
# ---------------------------------------------------------------------------

def _safe_globals() -> Dict:
    safe_builtins = {name: getattr(builtins, name)
                     for name in _ALLOWED_BUILTINS if hasattr(builtins, name)}
    return {
        "__builtins__": safe_builtins,
        # 不暴露 np —— 槽代码只许用白名单内建 + ctx。np 逻辑都封装在 ctx 原语里。
        "list": list, "range": range, "len": len,
        "int": int, "float": float, "min": min, "max": max, "abs": abs,
        "sum": sum, "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
        "zip": zip, "round": round, "all": all, "any": any,
        "set": set, "dict": dict, "tuple": tuple, "bool": bool,
    }


def compile_slots(slots: Dict[str, str], ctx) -> Tuple[bool, Dict[str, Callable], str]:
    """Compile + validate every slot. ctx is used for behavior checks.

    Returns (ok, {slot_name: fn}, message). On any failure ok=False and the
    returned dict may be partial.
    """
    fns: Dict[str, Callable] = {}
    for name, src in slots.items():
        if name not in SLOT_SIGNATURES:
            return False, fns, f"unknown slot: {name}"
        ok, reason = static_check(src)
        if not ok:
            return False, fns, f"[{name}] {reason}"
        g = _safe_globals()
        try:
            code = compile(src, f"<slot_{name}>", "exec")
            exec(code, g)
        except Exception as e:
            return False, fns, f"[{name}] compile/exec error: {type(e).__name__}: {e}"
        fn = g.get(name)
        if not callable(fn):
            return False, fns, f"[{name}] no callable '{name}' defined"
        ok, reason = _behavioral_check(fn, name, ctx)
        if not ok:
            return False, fns, f"[{name}] {reason}"
        fns[name] = fn
    return True, fns, "ok"


def _extract_json_object(text: str) -> str:
    """Try to pull a JSON object out of an LLM reply (strip code fences)."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # find first { ... last }
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        return text[i:j + 1]
    return text.strip()
