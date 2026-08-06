"""
Timer: Context manager for performance timing.
"""

import time
from contextlib import contextmanager


@contextmanager
def Timer(name: str = "Operation"):
    """
    Context manager that prints elapsed time.

    Usage:
        with Timer("Decoding"):
            solution = decoder.decode(permutation)

    Returns
    -------
    float
        Elapsed time in seconds (accessible via .elapsed attribute).
    """
    start = time.perf_counter()
    result = {"elapsed": 0.0}

    class TimerContext:
        elapsed = 0.0

    ctx = TimerContext()
    yield ctx
    elapsed = time.perf_counter() - start
    ctx.elapsed = elapsed
