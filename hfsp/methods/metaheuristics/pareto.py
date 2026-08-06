"""
Shared utilities for multi-objective optimization:
  - Non-dominated sorting (Deb et al., 2002)
  - Crowding distance computation
  - Weight vector generation via simplex lattice (Das & Dennis, 1998)
  - Tchebycheff decomposition scalarization
  - Hypervolume computation (2D)
  - Knee point / ideal-point selection helpers
"""

from typing import List, Optional
import numpy as np
from ...core.solution import ScheduleSolution


def compute_objective_matrix(
    solutions: List[ScheduleSolution],
    objective_names: tuple = ("makespan", "flow_time"),
) -> np.ndarray:
    """
    Extract objective values from a list of solutions.

    Parameters
    ----------
    solutions : list[ScheduleSolution]
    objective_names : tuple of str
        Which solution attributes to use as objectives.

    Returns
    -------
    np.ndarray of shape (len(solutions), len(objective_names))
    """
    n = len(solutions)
    m = len(objective_names)
    mat = np.zeros((n, m))
    for i, sol in enumerate(solutions):
        for j, name in enumerate(objective_names):
            mat[i, j] = getattr(sol, name, 0.0)
    return mat


def non_dominated_sort(
    obj_matrix: np.ndarray,
) -> List[List[int]]:
    """
    Deb's O(M·N²) non-dominated sorting (NSGA-II, 2002).

    For a minimization problem, solution i dominates solution j iff:
      - obj_matrix[i] <= obj_matrix[j] on ALL objectives, AND
      - obj_matrix[i] <  obj_matrix[j] on at least ONE objective.

    Parameters
    ----------
    obj_matrix : np.ndarray of shape (P, M)
        Objective values. P = population size, M = number of objectives.
        All objectives are assumed to be MINIMIZED.

    Returns
    -------
    fronts : list[list[int]]
        fronts[0] = indices of non-dominated (rank-0) solutions,
        fronts[1] = rank-1, etc.
    """
    p = len(obj_matrix)
    if p == 0:
        return []

    # S[i] = set of indices that solution i dominates
    S = [set() for _ in range(p)]
    # n[i] = number of solutions that dominate i
    n_dom = [0] * p

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if _dominates(obj_matrix[i], obj_matrix[j]):
                S[i].add(j)
            elif _dominates(obj_matrix[j], obj_matrix[i]):
                n_dom[i] += 1

    # Front 0: solutions with n_dom[i] == 0
    fronts = []
    current_front = [i for i in range(p) if n_dom[i] == 0]
    fronts.append(current_front)

    # Build subsequent fronts
    front_idx = 0
    while fronts[front_idx]:
        next_front = []
        for i in fronts[front_idx]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    next_front.append(j)
        front_idx += 1
        fronts.append(next_front)

    # Remove trailing empty front
    if fronts and not fronts[-1]:
        fronts.pop()

    return fronts


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a dominates b (both are minimized)."""
    return bool(np.all(a <= b) and np.any(a < b))


def crowding_distance(
    front_indices: List[int],
    obj_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute crowding distance for solutions in a single Pareto front.

    Boundary solutions get distance = inf (always preferred).
    Interior solutions get the sum of normalized neighbor differences
    across all objectives.

    Parameters
    ----------
    front_indices : list[int]
        Indices of solutions belonging to this front.
    obj_matrix : np.ndarray of shape (P, M)

    Returns
    -------
    distances : np.ndarray of shape (P,)
        Crowding distance for every solution. Solutions not in this
        front get distance = 0.0.
    """
    p = len(obj_matrix)
    m = obj_matrix.shape[1]
    distances = np.zeros(p)

    front = np.array(front_indices)
    if len(front) <= 2:
        # Every solution in a tiny front is a boundary solution
        for idx in front:
            distances[idx] = float("inf")
        return distances

    front_obj = obj_matrix[front]  # (|front|, M)

    for obj_idx in range(m):
        # Sort front by this objective
        sorted_order = np.argsort(front_obj[:, obj_idx])
        sorted_indices = front[sorted_order]
        sorted_vals = front_obj[sorted_order, obj_idx]

        f_min = sorted_vals[0]
        f_max = sorted_vals[-1]
        denom = f_max - f_min
        if denom < 1e-12:
            continue  # all identical on this objective

        # Boundary: distance = inf
        distances[sorted_indices[0]] = float("inf")
        distances[sorted_indices[-1]] = float("inf")

        # Interior
        for k in range(1, len(front) - 1):
            distances[sorted_indices[k]] += (
                sorted_vals[k + 1] - sorted_vals[k - 1]
            ) / denom

    return distances


def generate_weight_vectors(
    n_obj: int,
    H: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate N weight vectors uniformly distributed on the unit simplex.

    Uses the Das & Dennis (1998) simplex-lattice method:
      N = C(H + M - 1, M - 1)

    Each weight vector w satisfies:
      sum(w) = 1,  w_i ∈ {0, 1/H, 2/H, ..., 1}

    Parameters
    ----------
    n_obj : int
        Number of objectives (M).
    H : int
        Number of divisions along each axis.
    rng : np.random.Generator
        Used only if vectors need to be subsampled (not in core generation).

    Returns
    -------
    np.ndarray of shape (N, n_obj)
    """
    if n_obj == 2:
        # Fast path: simple 1D line
        weights = np.zeros((H + 1, 2))
        weights[:, 0] = np.linspace(0, 1, H + 1)
        weights[:, 1] = 1.0 - weights[:, 0]
        return weights

    # General case: recursive generation
    def _gen_recursive(m, remaining, current):
        if m == 1:
            yield current + [remaining / H]
        else:
            for v in range(remaining + 1):
                yield from _gen_recursive(m - 1, remaining - v, current + [v / H])

    vectors = list(_gen_recursive(n_obj, H, []))
    return np.array(vectors)


def tchebycheff_decomposition(
    objectives: np.ndarray,
    weight_vector: np.ndarray,
    ideal_point: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    Tchebycheff scalarization for a single solution.

        g(x | w, z*) = max_{i=1..M} { w_i' * |f_i(x) - z*_i| }

    Weights of 0 are replaced by epsilon to avoid degenerate cases
    where a zero-weighted objective is never optimized.

    Lower values are better.

    Parameters
    ----------
    objectives : np.ndarray of shape (M,)
    weight_vector : np.ndarray of shape (M,)
    ideal_point : np.ndarray of shape (M,)
    epsilon : float
        Minimum effective weight.

    Returns
    -------
    float
    """
    w = np.maximum(weight_vector, epsilon)
    return float(np.max(w * np.abs(objectives - ideal_point)))


def compute_hypervolume(
    front_obj: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
) -> float:
    """
    Compute hypervolume of a 2D Pareto front approximation.

    For M > 2, returns 0.0 with a warning (full n-D hypervolume is
    expensive; use a dedicated library like pygmo for production use).

    Parameters
    ----------
    front_obj : np.ndarray of shape (P, M)
        Non-dominated objective vectors.
    reference_point : np.ndarray of shape (M,), optional
        Reference point. Defaults to nadir point + 10% margin.

    Returns
    -------
    float
    """
    m = front_obj.shape[1]
    if m != 2:
        import warnings
        warnings.warn(
            f"Hypervolume for M={m} > 2 is not implemented. Returning 0.0."
        )
        return 0.0

    if reference_point is None:
        nadir = np.max(front_obj, axis=0)
        reference_point = nadir * 1.1

    # Sort by first objective ascending
    sorted_idx = np.argsort(front_obj[:, 0])
    sorted_front = front_obj[sorted_idx]

    hv = 0.0
    prev_x = 0.0
    for i in range(len(sorted_front)):
        width = sorted_front[i, 0] - prev_x
        height = reference_point[1] - sorted_front[i:, 1].min()
        if height > 0:
            hv += width * height
        prev_x = sorted_front[i, 0]

    # Remaining region up to reference point
    width = reference_point[0] - prev_x
    if width > 0:
        hv += width * (reference_point[1] - sorted_front[:, 1].min())

    return hv


def select_closest_to_ideal(
    solutions: List[ScheduleSolution],
    obj_matrix: np.ndarray,
    ideal_point: Optional[np.ndarray] = None,
    nadir_point: Optional[np.ndarray] = None,
) -> ScheduleSolution:
    """
    Select the solution closest to the ideal point (knee point heuristic).

    Uses Euclidean distance in normalized objective space
    (normalized by nadir - ideal range).

    Parameters
    ----------
    solutions : list[ScheduleSolution]
        Candidate solutions (typically the Pareto front).
    obj_matrix : np.ndarray of shape (P, M)
    ideal_point : np.ndarray of shape (M,), optional
    nadir_point : np.ndarray of shape (M,), optional

    Returns
    -------
    ScheduleSolution (copy)
    """
    if not solutions:
        raise ValueError("Empty solution list.")

    if ideal_point is None:
        ideal_point = np.min(obj_matrix, axis=0)
    if nadir_point is None:
        nadir_point = np.max(obj_matrix, axis=0)

    # Normalize
    denom = nadir_point - ideal_point
    denom[denom < 1e-12] = 1.0
    normalized = (obj_matrix - ideal_point) / denom

    # Euclidean distance to origin (ideal point in normalized space)
    distances = np.sqrt(np.sum(normalized ** 2, axis=1))
    best_idx = int(np.argmin(distances))

    return solutions[best_idx].copy()
