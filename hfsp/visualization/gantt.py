"""
Gantt chart visualization for HFSP schedules.

Displays the schedule as a Gantt chart with machines on the y-axis
and time on the x-axis, colored by job or by stage.
"""

from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ..core.solution import ScheduleSolution


# Default color palette (colorblind-friendly)
DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#5254a3", "#6b6ecf", "#b5cf6b", "#cedb9c", "#e7cb94",
    "#e7ba52", "#bd9e39", "#ad494a", "#d6616b", "#e7969c",
    "#a55194", "#ce6dbd", "#de9ed6", "#3182bd", "#6baed6",
    "#9ecae1", "#e6550d", "#fd8d3c", "#fdae6b", "#31a354",
    "#74c476", "#a1d99b", "#756bb1", "#9e9ac8", "#bcbddc",
]


def plot_gantt(
    solution: ScheduleSolution,
    title: Optional[str] = None,
    figsize: tuple = (14, 8),
    color_by: str = "job",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a Gantt chart for a schedule solution.

    Parameters
    ----------
    solution : ScheduleSolution
    title : str, optional
        Chart title.
    figsize : tuple
        Figure size (width, height).
    color_by : str
        "job": color each job differently.
        "stage": color each stage differently.
    save_path : str, optional
        If provided, save figure to this path.
    show : bool
        If True, display the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    instance = solution.instance
    assignments = solution.assignments

    if not assignments:
        raise ValueError("No assignments to plot.")

    n_machines = instance.total_machines
    n_jobs = instance.num_jobs
    n_stages = instance.num_stages

    # Build per-machine schedule
    machine_ops = {m: [] for m in range(n_machines)}
    for a in assignments:
        machine_ops[a["machine"]].append(a)
    for m in machine_ops:
        machine_ops[m].sort(key=lambda a: a["start"])

    # Determine colors
    if color_by == "job":
        n_colors = n_jobs
        color_key = lambda a: a["job"]
        color_label = "Job"
    else:  # stage
        n_colors = n_stages
        color_key = lambda a: a["stage"]
        color_label = "Stage"

    colors = DEFAULT_COLORS * (1 + n_colors // len(DEFAULT_COLORS))

    # Determine max time
    max_time = max(a["end"] for a in assignments) if assignments else 0
    max_time = max_time * 1.02  # small padding

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each operation as a horizontal bar
    for m in range(n_machines):
        stage = instance.stage_of_machine(m)
        for a in machine_ops[m]:
            job = a["job"]
            start = a["start"]
            duration = a["end"] - a["start"]
            idx = color_key(a)
            color = colors[idx % len(colors)]

            bar = ax.barh(
                y=m,
                width=duration,
                left=start,
                height=0.7,
                color=color,
                edgecolor="black",
                linewidth=0.3,
                alpha=0.85,
            )

            # Add job label in the bar if wide enough
            if duration > max_time * 0.02:
                ax.text(
                    start + duration / 2,
                    m,
                    f"J{job}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                )

    # Style the chart
    ax.set_yticks(range(n_machines))
    y_labels = []
    for m in range(n_machines):
        s = instance.stage_of_machine(m)
        y_labels.append(f"M{m} (S{s})")
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Machine (Stage)", fontsize=11)

    if title is None:
        title = (
            f"HFSP Gantt Chart — {instance.name}  "
            f"| Makespan: {solution.makespan:.1f}"
        )
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Draw stage separators
    cumulative = 0
    for s, count in enumerate(instance.machines_per_stage):
        if s > 0:
            ax.axhline(y=cumulative - 0.5, color="black", linewidth=1.5, linestyle="--")
        cumulative += count

    ax.set_ylim(-0.8, n_machines - 0.2)
    ax.invert_yaxis()

    # Legend
    if color_by == "job":
        if n_jobs <= 20:
            handles = [
                mpatches.Patch(color=colors[j % len(colors)], label=f"Job {j}")
                for j in range(n_jobs)
            ]
            ax.legend(
                handles=handles,
                loc="upper right",
                fontsize=6,
                ncol=max(1, n_jobs // 15),
                title=color_label,
            )
    else:
        handles = [
            mpatches.Patch(color=colors[s % len(colors)], label=f"Stage {s}")
            for s in range(n_stages)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, title=color_label)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
