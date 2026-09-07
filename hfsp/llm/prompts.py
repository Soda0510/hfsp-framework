"""
Prompt templates for LLM-guided metaheuristic mutation.

The mutation prompt asks the LLM to improve a candidate AlgorithmSpec given
its measured performance.  The system prompt states the component pool as an
explicit whitelist so the model stays inside the executable design space.
"""

import json
from typing import Dict, Optional

from .representation import AlgorithmSpec

SYSTEM_PROMPT_BASE = """\
You are an expert at designing metaheuristic algorithms for the Hybrid Flow \
Shop Scheduling Problem (HFSP). Given a current candidate algorithm (a JSON \
configuration) and its performance feedback, you propose ONE improved \
configuration.

An algorithm configuration is a JSON object with ONLY these fields and values:

- population_mode: true (population-based, GA-style) or false
- use_destroy_repair: true (IG-style destroy & reconstruct) or false
- initializer: "neh", "spt", "lpt", "palmer", "cds", or "random"
- mutation_operators: non-empty list, each in ["swap","insert","inverse","scramble","block"]
- crossover_operators: list, each in ["ox","pmx","two_point"] (must be non-empty if population_mode is true)
- crossover_prob: number in [0.5, 1.0]
- mutation_prob: number in [0.05, 0.5]
- acceptance: "always", "only_better", "metropolis", "threshold", or "record_to_record"
- temperature: null, or a number (used only when acceptance is "metropolis")
- threshold: null, or a number (used only when acceptance is "threshold")
- cooling_rate: number in [0.9, 0.999]
- use_local_search: true or false
- local_search_iterations: integer in [20, 200]
- population_size: integer in [10, 100]
- max_iterations: integer >= 0 (0 means a pure constructive method)
- elite_size: integer in [1, 5] (must be < population_size)
- tournament_size: integer in [2, 5]
- destruction_size: null, or an integer in [2, 8]

Rules:
1. Output ONLY a single valid JSON object. No explanation, no markdown, no code fences.
2. Your proposed configuration MUST differ meaningfully from the current one.
3. Use the feedback: if the current algorithm performs poorly, change its \
structure (population_mode / use_destroy_repair / acceptance); if it performs \
well, refine the parameters instead.
4. Respect every constraint above exactly.
"""


# ---------------------------------------------------------------------------
# Component glossary (阶段7): what each value actually DOES.  The whitelist
# above names the components and their legal ranges but deliberately omits
# their semantics; this block grounds the model's choices in the actual
# implementations (hfsp/methods/*), instead of relying on the pretrained sense
# of canonical (but ambiguous) names like "block", "scramble" or
# "record_to_record".
# ---------------------------------------------------------------------------

COMPONENT_GLOSSARY = """\
How the fields combine into an algorithm (dispatch):
- population_mode=true -> a genetic algorithm (tournament selection + crossover + mutation + elitism).
- population_mode=false & use_destroy_repair=true -> an iterated greedy (remove d jobs, reinsert each at its best position).
- population_mode=false & use_destroy_repair=false -> a single-solution search (mutation + acceptance, SA/ILS-style).
- max_iterations=0 -> a pure constructive method (initializer only, no search).

Initializers (build the starting job permutation):
- "neh": Nawaz-Enscore-Ham — sort jobs by total processing time (descending), then greedily insert each job at its best position.
- "spt": shortest processing time first (total processing time ascending).
- "lpt": longest processing time first (total processing time descending).
- "palmer": Palmer's slope-index rule — a weighted-sum ordering of jobs by their stage processing times.
- "cds": Campbell-Dudek-Smith — multi-pass Johnson's two-machine rule (keeps the best of the s-1 sequences).
- "random": a random permutation.

Mutation operators (perturb a job permutation):
- "swap": swap two randomly chosen jobs.
- "insert": remove one job and reinsert it at another position.
- "inverse": reverse a random contiguous subsequence.
- "scramble": randomly shuffle a contiguous subsequence.
- "block": move a random contiguous block of jobs (length up to n/3) to a new position.

Crossover operators (combine two parent permutations):
- "ox": order crossover — copy a subsequence from parent 1, fill the rest in parent 2's order.
- "pmx": partially mapped crossover.
- "two_point": two-point crossover.

Acceptance criteria (whether a worse neighbor may replace the current one):
- "always": accept every candidate.
- "only_better": accept only a strictly better candidate (hill-climbing).
- "metropolis": accept a worse candidate with probability exp(-Δ/T); T cools by cooling_rate (simulated annealing).
- "threshold": accept if the makespan increase is at most `threshold` (threshold accepting).
- "record_to_record": accept if the new makespan is within (1+threshold) of the best-so-far (record-to-record travel).
"""


def build_system_prompt(with_semantics: bool = True) -> str:
    """System prompt for config mutation, optionally augmented with semantics.

    ``with_semantics=True`` appends the component glossary so the model knows
    what each value does (not just its legal range).  The ``False`` arm is the
    ablation baseline: names only, semantics left to the model's pretrained
    knowledge.
    """
    base = SYSTEM_PROMPT_BASE.rstrip()
    if not with_semantics:
        return base
    return base + "\n\nComponent glossary (what each value does):\n" + COMPONENT_GLOSSARY


# Default (module-level) prompt includes the glossary — the improved version.
# Existing importers (mutator via build_system_prompt, __init__, run_oneshot)
# get the semantic prompt unless they explicitly opt out.
SYSTEM_PROMPT = build_system_prompt(True)


def build_design_prompt() -> str:
    """Ask the LLM to design an HFSP metaheuristic FROM SCRATCH (one shot).

    Used by ablation F: isolates the LLM's pretrained prior from the
    evolutionary search — a fresh design (no parent, no feedback) vs the
    evolved algorithm.
    """
    return (
        "Design a metaheuristic algorithm for the Hybrid Flow Shop Scheduling "
        "Problem (HFSP) minimizing makespan, from scratch. Choose the structure "
        "switches, initializer, operators, acceptance criterion and parameters "
        "that you believe will perform well. Output ONLY the AlgorithmSpec JSON "
        "object (same schema as above)."
    )


def build_mutation_prompt(
    parent: AlgorithmSpec,
    fitness: float,
    reflection: Optional[str] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    rank: Optional[int] = None,
    pop_best: Optional[float] = None,
    mode: Optional[str] = None,
    insights: Optional[str] = None,
) -> str:
    """Build the user prompt for one mutation step.

    ``mode`` (阶段3): "light" -> change ONE discrete component only;
    "medium" -> adjust ONE numeric parameter / neighborhood order; None -> free.

    ``examples`` (B: few-shot) is a list of dicts:
        {"parent": <spec dict>, "child": <spec dict>,
         "parent_fitness": float, "child_fitness": float}
    showing real, successful improvements from the evolution log.
    """
    lines: List[str] = []
    if examples:
        lines.append("Reference examples of successful algorithm improvements "
                     "(parent configuration -> improved child configuration):")
        lines.append("")
        for i, ex in enumerate(examples):
            lines.append(f"Example {i + 1}:")
            lines.append(f"  parent (RPD {ex['parent_fitness']:+.2f}%): "
                         f"{json.dumps(ex['parent'])}")
            lines.append(f"  improved (RPD {ex['child_fitness']:+.2f}%): "
                         f"{json.dumps(ex['child'])}")
            lines.append("")
    lines.append("Current candidate algorithm (JSON):")
    lines.append(json.dumps(parent.to_dict(), indent=2))
    lines.append("")
    lines.append("Performance feedback:")
    lines.append(f"  - mean RPD vs best-known: {fitness:+.2f}%   (lower is better; "
                 "0% ties the best-known, negative beats it)")
    if rank is not None:
        lines.append(f"  - population rank: #{rank} (0 = best of this generation)")
    if pop_best is not None:
        lines.append(f"  - current best RPD this generation: {pop_best:+.2f}%")
    if reflection:
        lines.append(f"  - analysis: {reflection}")
    if insights:
        lines.append(f"  - accumulated insights from past generations: {insights}")
    lines.append("")
    if mode == "light":
        lines.append("Task: do a LIGHT mutation — change EXACTLY ONE discrete "
                     "component (an operator ID, the initializer, or the "
                     "acceptance) to another legal value. Do NOT touch numeric "
                     "parameters.")
    elif mode == "medium":
        lines.append("Task: do a MEDIUM mutation — adjust ONE numeric parameter "
                     "(within its range) OR change the neighborhood operator "
                     "order. Keep the overall structure.")
    else:
        lines.append("Propose an improved configuration.")
    lines.append("Output ONLY the JSON object.")
    return "\n".join(lines)


def build_crossover_prompt(
    parent_a: AlgorithmSpec,
    parent_b: AlgorithmSpec,
    fitness_a: float,
    fitness_b: float,
) -> str:
    """Build a crossover prompt: combine two parents into one child config."""
    lines = [
        "Two parent algorithm configurations (JSON):",
        f"A (RPD {fitness_a:+.2f}%): {json.dumps(parent_a.to_dict())}",
        f"B (RPD {fitness_b:+.2f}%): {json.dumps(parent_b.to_dict())}",
        "",
        "Task: perform CROSSOVER — produce ONE child configuration by taking "
        "each component/parameter from either parent A or B (prefer the one "
        "with better RPD where they differ). You may make at most ONE small "
        "adjustment. Output ONLY the JSON object.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reflection prompts (P4): the LLM critiques WHY a spec performed as it did,
# and the critique is fed back into the next mutation ("verbal gradients").
# ---------------------------------------------------------------------------

ARCHIVE_INSIGHT_SYSTEM = """\
You are an expert at explaining why a metaheuristic algorithm configuration \
works on the Hybrid Flow Shop Scheduling Problem (HFSP). Given a good \
configuration and its performance, give ONE SHORT insight (1-2 sentences): \
which design choice is the likely reason it performs well, and what general \
principle this suggests. Output only the insight text, no JSON.
"""


def build_archive_prompt(spec: AlgorithmSpec, fitness: float) -> str:
    return (
        "A configuration that performs well (JSON):\n"
        + json.dumps(spec.to_dict(), indent=2)
        + f"\n\nIts mean RPD vs best-known: {fitness:+.2f}% (lower is better).\n"
        "Give one short insight: which design choice is the likely reason it "
        "performs well, and the general principle it suggests."
    )


def build_heavy_mutation_prompt(spec: AlgorithmSpec) -> str:
    """重变异 (阶段5): ask the LLM to design a NEW local-search operator.

    The design is recorded as an insight / candidate artifact; executing
    LLM-written code is avoided (keeps the pipeline reproducible).
    """
    return (
        "Current algorithm configuration (JSON):\n"
        + json.dumps(spec.to_dict(), indent=2)
        + "\n\nTask: HEAVY mutation — design a NEW local-search operator "
          "that could improve this algorithm, and explain WHY (1-2 sentences). "
          "The operator may introduce a new neighborhood (block swap, k-opt "
          "variant, critical-path based move) or a new acceptance/search "
          "strategy. Output ONLY:\n"
          "{\"name\": <operator name>, \"idea\": <1-2 sentence design>, "
          "\"reason\": <why it should help>}"
    )


REFLECTION_SYSTEM = """\
You are an expert at analyzing why a metaheuristic algorithm configuration \
performs well or poorly on the Hybrid Flow Shop Scheduling Problem (HFSP). \
You give a SHORT critique (1-2 sentences): the most likely reason for the \
observed performance, and the single most promising direction to improve this \
configuration. Be concrete about which structural choices (population mode, \
destroy & reconstruct, acceptance criterion, operators, parameters) are the \
likely cause. Output only the critique text, no JSON, no preamble.
"""


def build_reflection_prompt(spec: AlgorithmSpec, fitness: float) -> str:
    """Build the user prompt for a reflection (critique) step."""
    return (
        "Algorithm configuration (JSON):\n"
        + json.dumps(spec.to_dict(), indent=2)
        + f"\n\nIts performance on HFSP: mean RPD = {fitness:+.2f}% "
          "(lower is better; 0% ties the best-known).\n"
        "Give a short critique of why it performs this way and the most "
        "promising improvement direction."
    )
