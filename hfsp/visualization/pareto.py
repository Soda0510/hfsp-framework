"""
Pareto front visualization: 2D scatter plots of non-dominated solutions
and multi-algorithm comparison charts.
"""

from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from ..core.solution import ScheduleSolution


def plot_pareto_front(
    solutions: List[ScheduleSolution],
    objective_labels: tuple = ("Makespan", "Flow Time"),
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    highlight_knee: bool = True,
    color: str = "#1f77b4",
    marker: str = "o",
    label: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a 2D Pareto front as a scatter plot.

    Parameters
    ----------
    solutions : list[ScheduleSolution]
        The non-dominated solutions.
    objective_labels : tuple of str
        Labels for (x, y) axes.
    title : str, optional
    figsize : tuple
    highlight_knee : bool
        If True, highlight the knee point (closest to ideal) with a star.
    color : str
        Marker color.
    marker : str
        Marker style.
    label : str, optional
        Legend label.
    save_path : str, optional
        If given, save the figure to this path.
    show : bool
        If True, display the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.array([s.makespan for s in solutions])
    y = np.array([s.flow_time for s in solutions])

    ax.scatter(x, y, c=color, marker=marker, s=60, alpha=0.8,
               edgecolors="black", linewidth=0.5, label=label, zorder=5)

    if highlight_knee and len(solutions) > 0:
        # Knee point = closest to ideal (min on both axes)
        ideal = np.array([x.min(), y.min()])
        nadir = np.array([x.max(), y.max()])
        denom = nadir - ideal
        denom[denom < 1e-12] = 1.0
        normalized = np.column_stack([(x - ideal[0]) / denom[0],
                                       (y - ideal[1]) / denom[1]])
        distances = np.sqrt(np.sum(normalized ** 2, axis=1))
        knee_idx = int(np.argmin(distances))

        ax.scatter([x[knee_idx]], [y[knee_idx]], c="red", marker="*",
                   s=250, edgecolors="black", linewidth=0.5,
                   zorder=10, label="Knee point")

        # Ideal point
        ax.scatter([ideal[0]], [ideal[1]], c="green", marker="x",
                   s=100, linewidth=2, zorder=8, label="Ideal point")

    ax.set_xlabel(objective_labels[0], fontsize=12)
    ax.set_ylabel(objective_labels[1], fontsize=12)
    ax.set_title(title or "Pareto Front", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # Add annotation: front size
    ax.text(0.98, 0.02, f"Front size: {len(solutions)}",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_pareto_comparison(
    fronts: Dict[str, List[ScheduleSolution]],
    objective_labels: tuple = ("Makespan", "Flow Time"),
    title: Optional[str] = None,
    figsize: tuple = (10, 7),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Overlay multiple Pareto fronts for comparison.

    Parameters
    ----------
    fronts : dict[str, list[ScheduleSolution]]
        Mapping of algorithm name → list of Pareto-optimal solutions.
    objective_labels : tuple of str
    title : str, optional
    figsize : tuple
    save_path : str, optional
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(fronts), 3)))
    markers = ["o", "s", "D", "^", "v", "p"]

    for idx, (name, solutions) in enumerate(fronts.items()):
        x = np.array([s.makespan for s in solutions])
        y = np.array([s.flow_time for s in solutions])

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # Line connecting points to show front shape
        sorted_idx = np.argsort(x)
        ax.plot(x[sorted_idx], y[sorted_idx], "-", color=color,
                alpha=0.4, linewidth=1.5)

        ax.scatter(x, y, c=[color], marker=marker, s=70, alpha=0.8,
                   edgecolors="black", linewidth=0.5, label=name, zorder=5)

    ax.set_xlabel(objective_labels[0], fontsize=12)
    ax.set_ylabel(objective_labels[1], fontsize=12)
    ax.set_title(title or "Pareto Front Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    # Annotation: front sizes
    sizes = ", ".join(f"{k}: {len(v)}" for k, v in fronts.items())
    ax.text(0.98, 0.02, sizes,
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
