"""
HFSP: Hybrid Flow Shop Scheduling Problem Research Framework.

A modular, extensible framework for:
  - Modeling HFSP instances (core)
  - Exact solving via MILP (solvers)
  - Constructive heuristics and metaheuristics (methods)
  - Batch experiment management (experiment)
  - Visualization (visualization)

Quick start:
    from hfsp.io import InstanceReader
    from hfsp.methods.metaheuristics import GeneticAlgorithm

    instance = InstanceReader("Data").load("10-5-6")
    ga = GeneticAlgorithm(max_generations=200)
    solution = ga.solve(instance)
    print(f"Makespan: {solution.makespan:.1f}")
"""

__version__ = "0.1.0"
