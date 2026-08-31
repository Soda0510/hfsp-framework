"""
LLM-driven metaheuristic auto-design.

Turns "designing a metaheuristic algorithm" into a component-combination task:
an AlgorithmSpec (structured configuration) is assembled from existing
operators/heuristics and executed by ConfigurableMetaheuristic.  An LLM acts
as the mutation operator inside an evolutionary search over this design space.
"""

from .representation import AlgorithmSpec, spec_from_dict
from .components import validate_spec
from .executor import ConfigurableMetaheuristic
from .evaluate import make_fitness, per_instance_best
from .evolve import (
    EvolutionarySearch,
    random_spec,
    perturb_spec,
)
from .seeds import default_seed_specs
from .config import LLMConfig
from .client import LLMClient, OllamaClient
from .prompts import SYSTEM_PROMPT, build_mutation_prompt, build_design_prompt
from .mutator import LLMMutator
from .reflection import ReflectionEngine, DummyReflectionEngine

__all__ = [
    "AlgorithmSpec",
    "spec_from_dict",
    "validate_spec",
    "ConfigurableMetaheuristic",
    "make_fitness",
    "per_instance_best",
    "EvolutionarySearch",
    "random_spec",
    "perturb_spec",
    "default_seed_specs",
    "LLMConfig",
    "LLMClient",
    "OllamaClient",
    "SYSTEM_PROMPT",
    "build_mutation_prompt",
    "build_design_prompt",
    "LLMMutator",
    "ReflectionEngine",
    "DummyReflectionEngine",
]
