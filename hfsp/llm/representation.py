"""
AlgorithmSpec: structured representation of a candidate metaheuristic algorithm.

An algorithm is defined as a combination of components (initializer, mutation /
crossover operators, acceptance criterion) plus structural switches and
hyper-parameters.  The LLM operates on this structured object; the executor
(executor.py) turns it into a runnable ``Method``.

Keeping algorithms as *configurations* rather than raw code is what makes the
evolved algorithms safe, controllable and reproducible.
"""

from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional, Dict, Any


@dataclass
class AlgorithmSpec:
    """A candidate metaheuristic algorithm as a component combination.

    Structural switches choose the search template in the executor:

    - ``population_mode=True``      -> population-based, GA-style loop
    - ``population_mode=False`` and
      ``use_destroy_repair=True``   -> IG-style destroy & reconstruct loop
    - ``population_mode=False`` and
      ``use_destroy_repair=False``  -> single-solution search (SA/ILS-style)

    All fields have defaults so a spec is always constructible; validity of the
    values is checked against the component pool by ``components.validate_spec``.
    """

    name: str = "candidate"

    # ---- Structural switches ----
    population_mode: bool = True        # True -> population-based (GA-style)
    use_destroy_repair: bool = False    # True -> IG-style destroy & reconstruct

    # ---- Initialization ----
    initializer: str = "neh"            # "neh" | "spt" | "lpt" | "random"

    # ---- Operators ----
    mutation_operators: List[str] = field(
        default_factory=lambda: ["swap", "insert", "inverse"])
    crossover_operators: List[str] = field(
        default_factory=lambda: ["ox", "pmx", "two_point"])
    crossover_prob: float = 0.8
    mutation_prob: float = 0.3

    # ---- Acceptance criterion ----
    acceptance: str = "only_better"     # "always"|"only_better"|"metropolis"|"threshold"
    temperature: Optional[float] = None     # for "metropolis"
    threshold: Optional[float] = None       # for "threshold"
    cooling_rate: float = 0.97              # cooling for "metropolis" in single-solution mode

    # ---- Local search ----
    use_local_search: bool = False
    local_search_iterations: int = 100

    # ---- Scale ----
    population_size: int = 40
    max_iterations: int = 2000
    elite_size: int = 2
    tournament_size: int = 2
    destruction_size: Optional[int] = None  # for destroy-repair; None -> auto (max(2, n//10))

    # ---- Serialization helpers ----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"AlgorithmSpec(name={self.name!r}, population={self.population_mode}, "
            f"destroy={self.use_destroy_repair}, init={self.initializer}, "
            f"accept={self.acceptance}, iter={self.max_iterations})"
        )


# ---------------------------------------------------------------------------
# Parsing (LLM-friendly): build a validated spec from an arbitrary dict
# ---------------------------------------------------------------------------

_VALID_FIELDS = {f.name for f in fields(AlgorithmSpec)}

# Fields whose values must be coerced from whatever type the LLM emits.
_BOOL_FIELDS = {"population_mode", "use_destroy_repair", "use_local_search"}
_LIST_FIELDS = {"mutation_operators", "crossover_operators"}
_INT_FIELDS = {
    "local_search_iterations", "population_size", "max_iterations",
    "elite_size", "tournament_size", "destruction_size",
}
_FLOAT_FIELDS = {
    "crossover_prob", "mutation_prob", "temperature", "threshold", "cooling_rate",
}


def _coerce_value(field: str, value: Any) -> Any:
    """Coerce an LLM-provided value to the field's declared type."""
    if value is None:
        return None
    if field in _BOOL_FIELDS:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "1", "yes", "on")
    if field in _LIST_FIELDS:
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            parts = [p.strip().strip('"\'') for p in value.split(",") if p.strip()]
            return parts
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value]
        return [str(value)]
    if field in _INT_FIELDS:
        if isinstance(value, bool):
            raise ValueError(f"{field}: expected int, got bool")
        try:
            return int(float(value))
        except (TypeError, ValueError) as e:
            raise ValueError(f"{field}: cannot coerce {value!r} to int") from e
    if field in _FLOAT_FIELDS:
        if isinstance(value, bool):
            raise ValueError(f"{field}: expected float, got bool")
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{field}: cannot coerce {value!r} to float") from e
    return value  # name and other strings


def spec_from_dict(data: Dict[str, Any], name: str = "candidate") -> AlgorithmSpec:
    """Build a validated AlgorithmSpec from a dict (e.g. LLM output).

    - Unknown keys are ignored (LLMs often emit extra commentary fields).
    - Values are coerced to the declared field types (strings -> numbers/bools).
    - Cross-field inconsistencies are repaired, then validated; a ``ValueError``
      is raised if the resulting spec is still invalid.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict, got {type(data).__name__}")
    kwargs: Dict[str, Any] = {}
    for k in _VALID_FIELDS:
        if k in data:
            kwargs[k] = _coerce_value(k, data[k])
    if not kwargs.get("name"):
        kwargs["name"] = name

    from .components import repair_spec, validate_spec
    spec = repair_spec(AlgorithmSpec(**kwargs))
    errors = validate_spec(spec)
    if errors:
        raise ValueError(f"Invalid AlgorithmSpec: {'; '.join(errors)}")
    return spec
