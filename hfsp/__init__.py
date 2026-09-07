"""
HFSP: Hybrid Flow Shop Scheduling Problem Research Framework.

A modular, extensible framework for:
  - Modeling HFSP instances (core)
  - Exact solving via MILP (solvers)
  - Constructive heuristics and metaheuristics (methods)
  - LLM-guided metaheuristic auto-design (llm: component spec + code slots)
  - Visualization (visualization)

Instances live in benchmarks/seville/ (SevilleReader / SevilleReference).

Quick start:
    from hfsp.io import SevilleReader
    from hfsp.methods.metaheuristics import GeneticAlgorithm

    reader = SevilleReader("benchmarks/seville")
    instance = reader.load("instancia_10_10_1")
    ga = GeneticAlgorithm(max_generations=200)
    solution = ga.solve(instance)
    print(f"Makespan: {solution.makespan:.1f}")
"""

__version__ = "0.1.0"
