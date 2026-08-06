"""
Q-Learning Agent for adaptive operator selection in metaheuristics.

Integrates with GA to select crossover and mutation operators based on
population state (diversity, improvement rate, stagnation).

The agent maintains two Q-tables:
- One for crossover operator selection
- One for mutation operator selection

State is discretized from continuous population metrics.
"""

from typing import List, Optional
import numpy as np


class QLearningAgent:
    """
    Q-learning agent for adaptive operator selection.

    Parameters
    ----------
    n_operators : int
        Number of operators to choose from.
    n_states : int
        Number of discrete states.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration rate.
    epsilon_decay : float
        Multiplicative decay per episode.
    initial_q_table : np.ndarray, optional
        Pre-trained Q-table to initialize from.
    """

    def __init__(
        self,
        n_operators: int,
        n_states: int = 8,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        initial_q_table: Optional[np.ndarray] = None,
    ):
        self.n_operators = n_operators
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        if initial_q_table is not None and initial_q_table.shape == (n_states, n_operators):
            self.q_table = initial_q_table.copy().astype(float)
        else:
            self.q_table = np.zeros((n_states, n_operators), dtype=float)

        self.last_state = 0
        self.last_action = 0

    def select_action(self, state: int, rng: np.random.Generator) -> int:
        """Epsilon-greedy action selection. Returns operator index."""
        if rng.random() < self.epsilon:
            action = int(rng.integers(0, self.n_operators))
        else:
            action = int(np.argmax(self.q_table[state]))

        self.last_state = state
        self.last_action = action
        return action

    def update(self, reward: float, next_state: int):
        """Q-learning update after observing reward."""
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[self.last_state, self.last_action]
        self.q_table[self.last_state, self.last_action] += self.alpha * td_error

    def decay_epsilon(self):
        """Apply epsilon decay after each episode."""
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def get_q_table(self) -> np.ndarray:
        """Export the learned Q-table."""
        return self.q_table.copy()


def compute_state(
    diversity: float,
    improvement_rate: float,
    stagnation: int,
    n_states: int = 8,
) -> int:
    """
    Map continuous population metrics to a discrete Q-state.

    State encoding (3 bits):
    - diversity:   low(0) / medium(1) / high(2)  → 2 bits
    - improvement: none(0) / small(1) / large(2) → 2 bits
    - stagnation:  low(0) / high(1)               → 1 bit

    Returns state index in [0, n_states-1].
    """
    # Diversity: 0=low, 1=medium, 2=high
    if diversity < 0.3:
        div_code = 0
    elif diversity < 0.7:
        div_code = 1
    else:
        div_code = 2

    # Improvement rate
    if improvement_rate <= 0.0:
        imp_code = 0
    elif improvement_rate < 0.01:
        imp_code = 1
    else:
        imp_code = 2

    # Stagnation
    stag_code = 1 if stagnation > 10 else 0

    # Combine: div(2 bits=0..2) + imp(2 bits=0..2)*3 + stag(1 bit)*9
    state = div_code + imp_code * 3 + stag_code * 9
    return min(state, n_states - 1)


def compute_diversity(population: List) -> float:
    """
    Compute normalized population diversity (0..1).

    Uses average pairwise Hamming distance of permutations.
    """
    if len(population) < 2:
        return 0.0

    n = len(population[0].permutation)
    total_dist = 0.0
    count = 0
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            p1 = population[i].permutation
            p2 = population[j].permutation
            dist = sum(1 for a, b in zip(p1, p2) if a != b) / n
            total_dist += dist
            count += 1

    if count == 0:
        return 0.0
    return min(1.0, total_dist / count)
