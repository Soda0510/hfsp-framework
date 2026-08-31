"""
LLM-guided mutation operator.

Implements the same Mutator interface as ``perturb_spec`` (random mutation) so
the evolutionary loop in ``evolve.py`` is unchanged — the two differ only in
how offspring are proposed, which is exactly the paper's key ablation.

The LLM receives: parent spec (JSON) + its measured fitness (+ optional
reflection), and returns an improved spec.  Output is coerced + repaired +
validated; invalid outputs are retried up to ``max_retries`` times, and on
final failure the parent is kept (mutation no-op).
"""

from typing import Any, Dict, Optional
import numpy as np

from .representation import AlgorithmSpec, spec_from_dict
from .client import LLMClient
from .prompts import SYSTEM_PROMPT, build_mutation_prompt


class LLMMutator:
    """Mutate AlgorithmSpecs by asking a local LLM for an improved version."""

    def __init__(
        self,
        client: LLMClient,
        use_feedback: bool = True,
        max_retries: int = 3,
        temperature: Optional[float] = None,
        verbose: bool = True,
        examples: Optional[list] = None,
    ):
        self.client = client
        self.use_feedback = use_feedback    # False -> ablation arm "no feedback"
        self.max_retries = max_retries
        self.temperature = temperature
        self.verbose = verbose
        # B: few-shot examples of successful improvements (from evolution log).
        self.examples = examples or []
        # Tracking for analysis.
        self.stats = {"calls": 0, "failed": 0, "fell_back": 0}

    def mutate(self, parent: AlgorithmSpec,
               rng: Optional[np.random.Generator] = None,
               name: Optional[str] = None,
               feedback: Optional[Dict[str, Any]] = None) -> AlgorithmSpec:
        """Return a mutated spec; keeps the parent if the LLM output is unusable."""
        self.stats["calls"] += 1
        target = name or parent.name

        fitness = feedback.get("fitness") if (self.use_feedback and feedback) else None
        reflection = feedback.get("reflection") if self.use_feedback else None
        rank = feedback.get("rank") if (self.use_feedback and feedback) else None
        pop_best = feedback.get("pop_best") if (self.use_feedback and feedback) else None
        mode = feedback.get("mode") if (self.use_feedback and feedback) else None
        insights = feedback.get("insights") if (self.use_feedback and feedback) else None

        user = build_mutation_prompt(parent, fitness, reflection, examples=self.examples,
                                     rank=rank, pop_best=pop_best, mode=mode,
                                     insights=insights)
        last_err = None

        for attempt in range(self.max_retries):
            try:
                raw = self.client.chat_json(SYSTEM_PROMPT, user,
                                            temperature=self.temperature)
                child = spec_from_dict(raw, name=target)
                return child
            except Exception as e:  # network, JSON parse, coercion, validation
                last_err = e
                self.stats["failed"] += 1
                if self.verbose:
                    print(f"    [LLMMutator] attempt {attempt + 1}/{self.max_retries} "
                          f"failed ({type(e).__name__}: {str(e)[:80]})")

        # Fall back to the parent (mutation no-op) rather than an invalid spec.
        self.stats["fell_back"] += 1
        if self.verbose:
            print(f"    [LLMMutator] giving up after {self.max_retries} attempts "
                  f"({last_err}) — keeping parent '{parent.name}'")
        spec = AlgorithmSpec(**parent.to_dict())
        spec.name = target
        return spec

    def heavy_mutate(self, parent: AlgorithmSpec,
                     rng: Optional[np.random.Generator] = None,
                     name: Optional[str] = None,
                     feedback: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """重变异 (阶段5): ask the LLM to design a NEW local-search operator.

        Returns a design dict {name, idea, reason} (not a spec) — the operator
        design is recorded as an archive insight, not executed.
        """
        from .prompts import build_heavy_mutation_prompt
        self.stats["calls"] += 1
        try:
            raw = self.client.chat_json(SYSTEM_PROMPT, build_heavy_mutation_prompt(parent),
                                        temperature=self.temperature)
            if isinstance(raw, dict) and raw.get("name"):
                return raw
        except Exception as e:
            self.stats["failed"] += 1
            if self.verbose:
                print(f"    [LLMMutator heavy] failed: {str(e)[:80]}")
        return None

    def crossover(self, parent_a: AlgorithmSpec, parent_b: AlgorithmSpec,
                  rng: Optional[np.random.Generator] = None,
                  name: Optional[str] = None,
                  feedback: Optional[Dict[str, Any]] = None) -> AlgorithmSpec:
        """Combine two parents into one child (阶段3 crossover)."""
        from .prompts import build_crossover_prompt
        self.stats["calls"] += 1
        target = name or "crossover_child"
        fa = feedback.get("fitness_a") if feedback else None
        fb = feedback.get("fitness_b") if feedback else None
        user = build_crossover_prompt(
            parent_a, parent_b,
            fa if fa is not None else 0.0,
            fb if fb is not None else 0.0)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                raw = self.client.chat_json(SYSTEM_PROMPT, user,
                                            temperature=self.temperature)
                return spec_from_dict(raw, name=target)
            except Exception as e:
                last_err = e
                self.stats["failed"] += 1
                if self.verbose:
                    print(f"    [LLMMutator crossover] attempt {attempt + 1} failed: "
                          f"{str(e)[:80]}")
        self.stats["fell_back"] += 1
        # fall back to the better parent
        better = parent_a if fa is None or fb is None or fa <= fb else parent_b
        spec = AlgorithmSpec(**better.to_dict())
        spec.name = target
        return spec
