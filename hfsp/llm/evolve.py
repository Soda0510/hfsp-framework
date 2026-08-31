"""
Evolutionary search over the AlgorithmSpec design space.

The mutation operator is pluggable, so the same loop can be driven by:
  - random mutation (P2): the baseline loop, and the "random" arm of the
    paper's key ablation;
  - LLM mutation (P3): the actual research contribution.

Comparing the two under an identical loop is the central ablation question:
"is the LLM's mutation better than random mutation?"
"""

import json
import multiprocessing as mp
import time
from typing import Callable, List, Optional, Dict, Any
import numpy as np

from .representation import AlgorithmSpec
from .components import validate_spec, repair_spec, MUTATION_OPERATORS, CROSSOVER_OPERATORS
from .diversity import max_similarity


def _signature(spec: AlgorithmSpec) -> str:
    """Structural fingerprint of a spec (ignores the name) for fitness caching."""
    d = spec.to_dict()
    d.pop("name", None)
    return json.dumps(d, sort_keys=True, default=str)


# Module-level hook so fork-inherited pool workers can call the (closure) fitness
# without pickling it: the script sets this global before forking the pool.
_WORKER_FITNESS: Optional[Callable[[AlgorithmSpec], float]] = None


def _worker_fitness(spec: AlgorithmSpec) -> float:
    assert _WORKER_FITNESS is not None, "_WORKER_FITNESS not set"
    return _WORKER_FITNESS(spec)


# ---------------------------------------------------------------------------
# Random sampling helpers (legal values for each spec field)
# ---------------------------------------------------------------------------

def _random_mutation_subset(rng) -> List[str]:
    names = list(MUTATION_OPERATORS.keys())
    k = int(rng.integers(1, len(names) + 1))
    return list(rng.choice(names, size=k, replace=False))


def _random_crossover_subset(rng) -> List[str]:
    names = list(CROSSOVER_OPERATORS.keys())
    k = int(rng.integers(0, len(names) + 1))
    return list(rng.choice(names, size=k, replace=False))


def _random_field_value(field: str, rng: np.random.Generator):
    """Return a random legal value for a spec field."""
    if field == "population_mode":
        return bool(rng.integers(0, 2))
    if field == "use_destroy_repair":
        return bool(rng.integers(0, 2))
    if field == "initializer":
        return str(rng.choice(["random", "neh", "spt", "lpt", "palmer", "cds"]))
    if field == "mutation_operators":
        return _random_mutation_subset(rng)
    if field == "crossover_operators":
        return _random_crossover_subset(rng)
    if field == "crossover_prob":
        return float(rng.uniform(0.5, 1.0))
    if field == "mutation_prob":
        return float(rng.uniform(0.05, 0.5))
    if field == "acceptance":
        return str(rng.choice(
            ["always", "only_better", "metropolis", "threshold", "record_to_record"]))
    if field == "temperature":
        return None if rng.random() < 0.5 else float(rng.uniform(0.05, 1.0))
    if field == "threshold":
        return None if rng.random() < 0.6 else float(rng.uniform(0.0, 3.0))
    if field == "cooling_rate":
        return float(rng.uniform(0.9, 0.999))
    if field == "use_local_search":
        return bool(rng.integers(0, 2))
    if field == "local_search_iterations":
        return int(rng.integers(20, 201))
    if field == "population_size":
        return int(rng.integers(10, 31))    # modest scale keeps evaluations fast
    if field == "max_iterations":
        return int(rng.integers(100, 301))  # modest budget keeps evaluations fast
    if field == "elite_size":
        return int(rng.integers(1, 5))
    if field == "tournament_size":
        return int(rng.integers(2, 5))
    if field == "destruction_size":
        return None if rng.random() < 0.5 else int(rng.integers(2, 9))
    raise ValueError(f"Unknown field: {field}")


def random_spec(rng: np.random.Generator, name: str = "candidate") -> AlgorithmSpec:
    """Sample a fully random (but valid) AlgorithmSpec."""
    spec = AlgorithmSpec(
        name=name,
        population_mode=_random_field_value("population_mode", rng),
        use_destroy_repair=_random_field_value("use_destroy_repair", rng),
        initializer=_random_field_value("initializer", rng),
        mutation_operators=_random_mutation_subset(rng),
        crossover_operators=_random_crossover_subset(rng),
        crossover_prob=_random_field_value("crossover_prob", rng),
        mutation_prob=_random_field_value("mutation_prob", rng),
        acceptance=_random_field_value("acceptance", rng),
        temperature=_random_field_value("temperature", rng),
        threshold=_random_field_value("threshold", rng),
        cooling_rate=_random_field_value("cooling_rate", rng),
        use_local_search=_random_field_value("use_local_search", rng),
        local_search_iterations=_random_field_value("local_search_iterations", rng),
        population_size=_random_field_value("population_size", rng),
        max_iterations=_random_field_value("max_iterations", rng),
        elite_size=_random_field_value("elite_size", rng),
        tournament_size=_random_field_value("tournament_size", rng),
        destruction_size=_random_field_value("destruction_size", rng),
    )
    repair_spec(spec)
    errors = validate_spec(spec)
    if errors:
        raise RuntimeError(f"random_spec produced invalid spec: {errors}")
    return spec


def perturb_spec(parent: AlgorithmSpec,
                 rng: np.random.Generator,
                 name: Optional[str] = None,
                 feedback: Optional[Dict[str, Any]] = None) -> AlgorithmSpec:
    """Mutate a parent spec by re-randomizing 1-3 randomly chosen fields.

    ``feedback`` is ignored by random mutation; it exists so the mutator
    signature is uniform with the LLM mutator (P3).
    """
    spec = AlgorithmSpec(**parent.to_dict())
    if name:
        spec.name = name

    n_changes = int(rng.integers(1, 4))
    fields = [
        "population_mode", "use_destroy_repair", "initializer",
        "mutation_operators", "crossover_operators", "acceptance",
        "crossover_prob", "mutation_prob", "use_local_search",
        "population_size", "max_iterations", "temperature", "threshold",
        "cooling_rate", "local_search_iterations", "elite_size",
        "tournament_size", "destruction_size",
    ]
    chosen = rng.choice(fields, size=n_changes, replace=False)
    for f in chosen:
        setattr(spec, f, _random_field_value(f, rng))

    repair_spec(spec)
    errors = validate_spec(spec)
    if errors:
        # Fall back to a fresh random spec rather than propagating an invalid one.
        return random_spec(rng, name or "candidate")
    return spec


# ---------------------------------------------------------------------------
# Evolutionary search loop
# ---------------------------------------------------------------------------

# A mutator is a callable: mutator(parent, rng, name, feedback) -> AlgorithmSpec.
# `feedback` is an optional dict (e.g. {"fitness": parent_fitness}) that guides
# LLM mutation; random mutation ignores it.
Mutator = Callable[[AlgorithmSpec, np.random.Generator, str, Optional[Dict[str, Any]]], AlgorithmSpec]


class EvolutionarySearch:
    """Population-based evolution over the AlgorithmSpec design space."""

    def __init__(
        self,
        fitness_fn: Callable[[AlgorithmSpec], float],
        mutator: Mutator,
        population_size: int = 8,
        elite_size: int = 2,
        generations: int = 5,
        seed: int = 0,
        seed_specs: Optional[List[AlgorithmSpec]] = None,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = True,
        use_reflection: bool = False,
        reflection_engine: Optional["ReflectionEngine"] = None,
        log_mutations: bool = False,
        n_jobs: int = 1,
        crossover_mutator: Optional[Callable] = None,
        mutation_ratios: tuple = (0.3, 0.6, 0.1),  # (light, medium, crossover)
        diversity_lambda: float = 0.0,             # 阶段4 anti-collapse penalty
        archive: Optional["InsightArchive"] = None,# 阶段5 cross-gen memory
    ):
        self.fitness_fn = fitness_fn
        self.mutator = mutator
        self.population_size = population_size
        self.elite_size = elite_size
        self.generations = generations
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.seed_specs = seed_specs or []
        self.verbose = verbose
        # ---- Reflection (P4): verbal gradients fed into mutation prompts ----
        self.use_reflection = use_reflection
        self.reflection_engine = reflection_engine
        # ---- Mutation log (for M8 fine-tuning data) ----
        self.log_mutations = log_mutations
        self.mutation_log: List[Dict[str, Any]] = []
        # ---- Parallel fitness evaluation ----
        self.n_jobs = n_jobs
        # ---- 阶段3: mutation modes + crossover ----
        self.crossover_mutator = crossover_mutator
        self.light_ratio, self.medium_ratio, self.crossover_ratio = mutation_ratios
        # ---- 阶段4: diversity penalty (anti-collapse) ----
        self.diversity_lambda = diversity_lambda
        # ---- 阶段5: insight archive (cross-gen memory) ----
        self.archive = archive

    def run(self) -> Dict[str, Any]:
        # ---- Initialize population: seeds + random fill ----
        population = list(self.seed_specs)
        while len(population) < self.population_size:
            population.append(random_spec(self.rng, "init"))

        history: List[float] = []
        history_mean: List[float] = []
        history_diversity: List[float] = []   # unique structural signatures / pop size
        best_spec, best_fitness = None, float("inf")
        t_start = time.perf_counter()

        # Fitness cache: structurally identical specs (e.g. preserved elites)
        # are evaluated only once.  This is essential when evaluation is the
        # expensive part (P3+).
        cache: Dict[str, float] = {}

        # Parallel evaluation pool (fork context).  The fitness is a closure
        # (not picklable), so we expose it via the module global _WORKER_FITNESS
        # which forked workers inherit; pool.map only pickles the specs.
        pool = None
        if self.n_jobs > 1:
            global _WORKER_FITNESS
            _WORKER_FITNESS = self.fitness_fn
            pool = mp.get_context("fork").Pool(self.n_jobs)

        def eval_population(population: List[AlgorithmSpec]) -> Dict[int, float]:
            """Evaluate all specs, parallelizing only the uncached ones."""
            uncached = [s for s in population if _signature(s) not in cache]
            if pool is not None and len(uncached) > 1:
                vals = pool.map(_worker_fitness, uncached)
                for s, v in zip(uncached, vals):
                    cache[_signature(s)] = v
            else:
                for s in uncached:
                    cache[_signature(s)] = self.fitness_fn(s)
            return {id(s): cache[_signature(s)] for s in population}

        for gen in range(self.generations + 1):
            # ---- Evaluate (parallel) ----
            fits = eval_population(population)
            if self.diversity_lambda > 0 and len(population) > self.elite_size:
                # 阶段4 anti-collapse: penalize non-elites similar to the elites.
                elites = population[:self.elite_size]
                eff = {id(s): fits[id(s)] for s in elites}
                for s in population:
                    if id(s) not in eff:
                        sim = max_similarity(s, elites)
                        eff[id(s)] = fits[id(s)] - self.diversity_lambda * sim
                population.sort(key=lambda s: eff[id(s)])
            else:
                population.sort(key=lambda s: fits[id(s)])

            if fits[id(population[0])] < best_fitness:
                best_fitness = fits[id(population[0])]
                best_spec = population[0]
            history.append(best_fitness)
            history_mean.append(float(np.mean([fits[id(s)] for s in population])))
            # 阶段4: per-generation population diversity (unique structures / size)
            unique = len({_signature(s) for s in population})
            history_diversity.append(unique / max(len(population), 1))

            if self.verbose:
                elapsed = time.perf_counter() - t_start
                per = elapsed / max(gen + 1, 1)
                eta = per * (self.generations - gen)
                print(f"  gen {gen:>2}/{self.generations}  best RPD = {best_fitness:+.2f}%  "
                      f"({best_spec.name})  [{elapsed/60:>5.1f}min, "
                      f"{per/60:.1f}min/gen, ETA {eta/60:.0f}min]  | top5: "
                      f"{[round(fits[id(s)], 2) for s in population[:5]]}")

            # ---- Reflection (P4): critique the worst spec this generation ----
            if (self.use_reflection and self.reflection_engine is not None
                    and len(population) > 0):
                self.reflection_engine.get_or_reflect(
                    population[-1], fits[id(population[-1])])

            # ---- Archive insight (阶段5): why the best works (cross-gen memory) ----
            if self.archive is not None and len(population) > 0:
                self.archive.ask_and_record(population[0], fits[id(population[0])])

            if gen == self.generations:
                break

            # ---- Selection + offspring ----
            new_pop = population[:self.elite_size]
            parent_pool = population[:max(3, self.population_size // 2)]
            # Rank feedback (阶段1): feed the LLM the population RANK (0 = best)
            # + relative distance to the current best, not a bare makespan.
            ranks = {id(s): i for i, s in enumerate(population)}
            pop_best = fits[id(population[0])]
            while len(new_pop) < self.population_size:
                name = f"g{gen + 1}"
                roll = self.rng.random()
                op_mode = "crossover"
                if (self.crossover_mutator is not None
                        and roll >= (self.light_ratio + self.medium_ratio)):
                    # ---- crossover: combine two parents (阶段3) ----
                    parent = parent_pool[int(self.rng.integers(0, len(parent_pool)))]
                    parent2 = parent_pool[int(self.rng.integers(0, len(parent_pool)))]
                    feedback = {"fitness_a": fits[id(parent)],
                                "fitness_b": fits[id(parent2)]}
                    child = self.crossover_mutator(parent, parent2, self.rng, name, feedback)
                else:
                    # ---- single-parent mutation, light / medium by ratio ----
                    op_mode = "light" if roll < self.light_ratio else "medium"
                    parent = parent_pool[int(self.rng.integers(0, len(parent_pool)))]
                    feedback: Dict[str, Any] = {
                        "fitness": fits[id(parent)],
                        "rank": ranks[id(parent)],
                        "pop_best": pop_best,
                        "mode": op_mode,
                    }
                    if self.use_reflection and self.reflection_engine is not None:
                        feedback["reflection"] = self.reflection_engine.get_or_reflect(
                            parent, fits[id(parent)])
                    if self.archive is not None:
                        s = self.archive.summary()
                        if s:
                            feedback["insights"] = s
                    child = self.mutator(parent, self.rng, name, feedback)
                if child is None:
                    child = random_spec(self.rng, name)
                if self.log_mutations:
                    self.mutation_log.append({
                        "gen": gen + 1,
                        "parent": parent.to_dict(),
                        "child": child.to_dict(),
                        "parent_fitness": float(fits[id(parent)]),
                        "mode": op_mode,
                    })
                new_pop.append(child)
            population = new_pop

        if pool is not None:
            pool.close()
            pool.join()

        return {
            "best_spec": best_spec,
            "best_fitness": best_fitness,
            "history": history,          # best RPD per generation
            "history_mean": history_mean,# population-mean RPD per generation
            "history_diversity": history_diversity,  # unique structures / pop size
        }
