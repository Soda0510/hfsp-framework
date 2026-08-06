"""
RNGManager: Centralized random number generation with seed management
for reproducible experiments.
"""

import numpy as np


class RNGManager:
    """
    Manages random number generators with deterministic seeding.

    Usage:
        rng_mgr = RNGManager(base_seed=42)
        rng = rng_mgr.get_generator(run_id=0, stream="sampling")
    """

    def __init__(self, base_seed: int = 20240101):
        self.base_seed = base_seed

    def get_generator(self, run_id: int = 0, stream: str = "default") -> np.random.Generator:
        """
        Create a seeded random generator.

        Parameters
        ----------
        run_id : int
            Experiment run ID.
        stream : str
            Named stream for different RNG purposes (e.g., "sampling", "operator").

        Returns
        -------
        np.random.Generator
        """
        stream_hash = hash(stream) & 0x7FFFFFFF
        seed = self.base_seed + run_id * 1000 + stream_hash
        return np.random.Generator(np.random.PCG64(seed))
