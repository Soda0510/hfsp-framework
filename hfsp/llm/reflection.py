"""
Reflection engine (P4): asks the LLM to critique WHY a spec performed as it
did, and feeds the critique back into the next mutation prompt.

This mirrors ReEvo's "verbal gradients": instead of only giving the LLM a
scalar fitness, we give it a short natural-language analysis of the candidate,
so the next mutation knows what went wrong and what to change.

The ablation "reflection vs no-reflection" simply toggles whether a
``ReflectionEngine`` is attached to the evolutionary loop.
"""

from typing import Optional

from .representation import AlgorithmSpec
from .client import LLMClient
from .prompts import REFLECTION_SYSTEM, build_reflection_prompt


def _sig(spec: AlgorithmSpec) -> str:
    from .evolve import _signature  # lazy import to avoid a module-load cycle
    return _signature(spec)


class ReflectionEngine:
    """Caches per-spec critiques and accumulates short-term lessons."""

    def __init__(self, client: LLMClient, verbose: bool = True):
        self.client = client
        self.verbose = verbose
        self._cache = {}          # spec signature -> critique text
        self.lessons: list[str] = []  # most recent critiques (long-term context)
        self.max_lessons = 8
        self.stats = {"calls": 0, "failed": 0}

    # ------------------------------------------------------------------
    # API used by the evolutionary loop
    # ------------------------------------------------------------------

    def get_or_reflect(self, spec: AlgorithmSpec, fitness: float) -> str:
        """Return a cached critique for this spec, or generate + cache one."""
        key = _sig(spec)
        if key not in self._cache:
            text = self._reflect(spec, fitness)
            if text:
                self._cache[key] = text
                self.lessons.append(text)
                if len(self.lessons) > self.max_lessons:
                    self.lessons.pop(0)
        return self._cache.get(key, "")

    def get(self, spec: AlgorithmSpec) -> str:
        """Return a previously cached critique (without calling the LLM)."""
        return self._cache.get(_sig(spec), "")

    def lesson_summary(self) -> str:
        """Concatenated recent critiques — optional long-term guidance."""
        return " ".join(self.lessons)

    # ------------------------------------------------------------------

    def _reflect(self, spec: AlgorithmSpec, fitness: float) -> Optional[str]:
        try:
            self.stats["calls"] += 1
            text = self.client.chat_text(
                REFLECTION_SYSTEM, build_reflection_prompt(spec, fitness))
            text = text.strip().strip('"')
            if not text:
                return None
            return text[:500]   # cap length; critiques should be short anyway
        except Exception as e:
            self.stats["failed"] += 1
            if self.verbose:
                print(f"    [ReflectionEngine] call failed: {str(e)[:80]}")
            return None


class DummyReflectionEngine(ReflectionEngine):
    """Deterministic stand-in for tests / ablation sanity (no LLM calls)."""

    def __init__(self):
        self._cache = {}
        self.lessons = []
        self.max_lessons = 8
        self.stats = {"calls": 0, "failed": 0}

    def get_or_reflect(self, spec: AlgorithmSpec, fitness: float) -> str:
        return f"[dummy] fitness {fitness:+.2f}% — likely caused by structure; try changing it."
