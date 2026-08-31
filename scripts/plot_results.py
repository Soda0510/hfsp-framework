"""
P5e: visualize the evolution convergence and the test-set RPD comparison.

Produces (saved to results/llm/figures/):
    fig1_evolution_convergence.png   best/mean RPD vs generation + seed baselines
    fig2_rpd_comparison.png          boxplot of RPD per algorithm on the test set

Usage:
    .venv/bin/python scripts/plot_results.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("results/llm/base_run")
P5D = Path("results/llm/p5d")
FIG = Path("results/llm/figures")


def plot_convergence():
    """Best/mean RPD over generations, with seed baselines as reference lines."""
    curve = pd.read_csv(BASE / "evolution_curve.csv")
    seeds = pd.read_csv(BASE / "seed_baselines.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["generation"], curve["best_rpd"], "o-", color="#C0392B",
            label="best (evolved)", linewidth=1.6, markersize=4)
    ax.plot(curve["generation"], curve["mean_rpd"], "s--", color="#2980B9",
            label="population mean", linewidth=1.3, markersize=3)
    # seed baselines
    for _, r in seeds.iterrows():
        ax.axhline(r["rpd"], color="#7F8C8D", linestyle=":", linewidth=1.0,
                   alpha=0.8)
        ax.text(curve["generation"].max() + 0.3, r["rpd"], f"{r['seed']}",
                fontsize=8, color="#7F8C8D", va="center")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean RPD vs UpperBounds (%)")
    ax.set_title("Base-model evolution convergence (training subset)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_evolution_convergence.png", dpi=150)
    plt.close(fig)
    print("saved fig1_evolution_convergence.png")


def plot_rpd_comparison():
    """Boxplot of per-instance RPD for each algorithm on the test set."""
    df = pd.read_csv(P5D / "rpd_table.csv")
    algos = list(df["algorithm"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df.loc[df["algorithm"] == a, "rpd"].values for a in algos]
    bp = ax.boxplot(data, tick_labels=algos, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "#C0392B",
                               "markeredgecolor": "#C0392B", "markersize": 5})
    for patch in bp["boxes"]:
        patch.set_facecolor("#AED6F1")
        patch.set_alpha(0.7)
    ax.set_ylabel("RPD vs UpperBounds (%)")
    ax.set_title("Test-set RPD by algorithm (144 held-out instances)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_rpd_comparison.png", dpi=150)
    plt.close(fig)
    print("saved fig2_rpd_comparison.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plot_convergence()
    if (P5D / "rpd_table.csv").exists():
        plot_rpd_comparison()
    else:
        print("p5d results not ready yet — skip fig2 (run after P5d completes)")


if __name__ == "__main__":
    main()
