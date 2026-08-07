#!/usr/bin/env python3
"""
Generate a comprehensive comparison chart from experiment results.
Output: a single figure with RPD bar chart + runtime comparison.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# 1. Load data
# ============================================================
results_path = "results/results.csv"
if not os.path.exists(results_path):
    print(f"ERROR: {results_path} not found. Run experiment first.")
    sys.exit(1)

df = pd.read_csv(results_path)

# ============================================================
# 2. Compute RPD (NEH as baseline per instance)
# ============================================================
neh_makespan = df[df["algorithm"] == "NEH"].groupby("instance")["makespan"].mean()
df["baseline"] = df["instance"].map(neh_makespan)
df["rpd"] = 100.0 * (df["makespan"] - df["baseline"]) / df["baseline"]

# ============================================================
# 3. Aggregate by instance group and algorithm
# ============================================================
def instance_group(name):
    parts = name.split("-")
    n_jobs = int(parts[0])
    return f"{n_jobs} jobs"

df["group"] = df["instance"].apply(instance_group)

# Per-instance RPD mean
rpd_by_instance = df[df["algorithm"] != "NEH"].groupby(
    ["instance", "algorithm"]
)["rpd"].mean().reset_index()

# Overall RPD by algorithm and job size
rpd_by_group = df[df["algorithm"] != "NEH"].groupby(
    ["group", "algorithm"]
)["rpd"].mean().reset_index()

# Runtime by algorithm and job size
runtime_by_group = df.groupby(["group", "algorithm"])["runtime"].mean().reset_index()

# ============================================================
# 4. Plot — single comprehensive figure
# ============================================================
algo_list = ["GA", "SA", "IG", "DPSO"]
algo_colors = {"GA": "#2ca02c", "SA": "#ff7f0e", "IG": "#d62728", "DPSO": "#1f77b4"}
group_order = ["10 jobs", "20 jobs", "50 jobs"]

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.30)

# ---- A. Overall RPD bar chart (grouped by job size) ----
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(len(group_order))
bar_width = 0.18
for i, algo in enumerate(algo_list):
    sub = rpd_by_group[rpd_by_group["algorithm"] == algo]
    vals = []
    for g in group_order:
        row = sub[sub["group"] == g]
        vals.append(row["rpd"].values[0] if len(row) > 0 else 0)
    bars = ax1.bar(x + (i - 1.5) * bar_width, vals, bar_width,
                   label=algo, color=algo_colors[algo], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val < 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.15,
                     f"{val:.1f}%", ha="center", va="top", fontsize=9, fontweight="bold", color="white")
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(group_order, fontsize=12)
ax1.set_ylabel("RPD vs NEH (%)", fontsize=12)
ax1.set_title("Average RPD by Problem Size (vs NEH baseline)", fontsize=14, fontweight="bold")
ax1.legend(fontsize=10, loc="lower left")
ax1.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
ax1.grid(axis="y", alpha=0.3)

# ---- B. Per-instance RPD heatmap-style ----
ax2 = fig.add_subplot(gs[0, 2])
# Pivot: rows=instance, cols=algorithm, values=rpd
pivot = rpd_by_instance.pivot(index="instance", columns="algorithm", values="rpd")
# Sort by instance name numerically
instance_order = sorted(pivot.index, key=lambda x: tuple(int(p) for p in x.split("-")))
pivot = pivot.loc[instance_order]

x_pos = np.arange(len(algo_list))
for i, inst in enumerate(instance_order):
    vals = [pivot.loc[inst, a] if a in pivot.columns and not pd.isna(pivot.loc[inst, a]) else 0 for a in algo_list]
    ax2.plot(x_pos, vals, "o-", linewidth=1.5, markersize=6, label=inst, alpha=0.8)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(algo_list, fontsize=11)
ax2.set_ylabel("RPD (%)", fontsize=11)
ax2.set_title("RPD by Instance", fontsize=13, fontweight="bold")
ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
ax2.grid(alpha=0.3)
# Legend outside
ax2.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1,
           title="Instance", title_fontsize=8)

# ---- C. Runtime comparison ----
ax3 = fig.add_subplot(gs[1, :2])
for i, algo in enumerate(algo_list):
    sub = runtime_by_group[runtime_by_group["algorithm"] == algo]
    vals = []
    for g in group_order:
        row = sub[sub["group"] == g]
        vals.append(row["runtime"].values[0] if len(row) > 0 else 0)
    ax3.plot(x, vals, "o-", linewidth=2.5, markersize=10, label=algo,
             color=algo_colors[algo], markerfacecolor="white", markeredgewidth=2)

ax3.set_xticks(x)
ax3.set_xticklabels(group_order, fontsize=12)
ax3.set_ylabel("Avg Runtime (s)", fontsize=12)
ax3.set_title("Average Runtime by Problem Size", fontsize=14, fontweight="bold")
ax3.legend(fontsize=11, loc="upper left")
ax3.grid(alpha=0.3)
ax3.set_yscale("log")

# ---- D. Summary text box ----
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis("off")

# Compute key stats
overall_rpd = df[df["algorithm"] != "NEH"].groupby("algorithm")["rpd"].mean()
best_algo = overall_rpd.idxmin()
best_rpd = overall_rpd.min()
fastest_algo = df.groupby("algorithm")["runtime"].mean().idxmin()
slowest_algo = df.groupby("algorithm")["runtime"].mean().idxmax()

summary_lines = [
    "KEY FINDINGS",
    "=" * 30,
    f"Instances: {df['instance'].nunique()} (10/20/50 jobs)",
    f"Algorithms: NEH(baseline) + 4",
    f"Runs per config: {df['run'].max() + 1}",
    "",
    "RPD vs NEH (avg over all instances):",
]
for algo in algo_list:
    v = overall_rpd.get(algo, 0)
    summary_lines.append(f"  {algo}: {v:+.1f}%")

summary_lines += [
    "",
    f"Best quality: {best_algo} ({best_rpd:+.1f}%)",
    f"Fastest: {fastest_algo}",
    f"Slowest: {slowest_algo}",
    "",
    "All experiments seed-managed",
    "for reproducibility.",
]

summary_text = "\n".join(summary_lines)
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=10, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

fig.suptitle("HFSP Algorithm Comparison — NEH vs GA / SA / IG / DPSO",
             fontsize=16, fontweight="bold", y=1.01)

# Save
output_path = "results/comparison_chart.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Chart saved to {output_path}")
print(f"\nOverall RPD vs NEH:\n{overall_rpd.to_string()}")
print(f"\nAvg Runtime:\n{df.groupby('algorithm')['runtime'].mean().to_string()}")
