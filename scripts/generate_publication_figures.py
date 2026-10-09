import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication-grade styling (Nature / Science / Cell)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.labelsize": 11.5,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
})

# 2-Row Layout to give plenty of breathing room and eliminate all text overlap:
# Top row: Panel A (Tertiary Fold Wins) and Panel B (Data Efficiency Frontier)
# Bottom row: Panel C (Full 8-Metric CASP15 Performance Delta Profile)
fig = plt.figure(figsize=(15, 11), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.05, 1.0], hspace=0.38, wspace=0.28)

c_af2 = "#64748b"      # Muted slate grey
c_nano = "#0d9488"     # Deep emerald / teal
c_accent = "#e11d48"   # Rose red for bottleneck
c_trailing = "#f59e0b" # Amber for moderate deficit

# ==============================================================
# PANEL A: Tertiary & Local Structural Discovery Wins (Top-Left)
# ==============================================================
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor("#ffffff")
ax_a.grid(axis="y", linestyle="--", color="#cbd5e1", zorder=0)

metrics_a = ["GDT_HA\n(CA Backbone)", "lDDT\n(All-Atom14)", "SphereGrinder\n(Local Env)", "Backbone\nDihedrals (BB)"]
af2_vals_a = [0.0611, 0.1530, 0.0837, 0.6042]
nano_vals_a = [0.1080, 0.2945, 0.3639, 0.6186]
delta_labels = ["+76.8%", "+92.5%", "4.34×\n(+334.8%)", "+2.4%"]

x = np.arange(len(metrics_a))
bar_width = 0.35

rects1 = ax_a.bar(x - bar_width/2, af2_vals_a, bar_width, label="minAlphaFold2 Full (94.2M)", color=c_af2, edgecolor="#1e293b", linewidth=1.1, zorder=3)
rects2 = ax_a.bar(x + bar_width/2, nano_vals_a, bar_width, label="NanoFold v1 (2.83M, Ours)", color=c_nano, edgecolor="#042f2e", linewidth=1.1, zorder=3)

for i in range(len(metrics_a)):
    y_pos = nano_vals_a[i] + 0.02
    ax_a.text(x[i] + bar_width/2, y_pos, delta_labels[i], ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0f766e")

ax_a.set_ylabel("Evaluation Score (0.0 to 1.0)", labelpad=8)
ax_a.set_title("A   Tertiary & Local Fold Discovery Wins", loc="left", fontweight="bold", pad=12, fontsize=13)
ax_a.set_xticks(x)
ax_a.set_xticklabels(metrics_a)
ax_a.set_ylim(0, 0.76)
ax_a.legend(loc="upper left", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", framealpha=0.95)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)


# ==============================================================
# PANEL B: Data Efficiency Frontier (Top-Right)
# ==============================================================
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor("#ffffff")
ax_b.grid(True, linestyle="--", color="#cbd5e1", zorder=0)

# Parameter comparison (Millions) vs GDT_HA
params = [2.83, 94.2, 1.4]
gdt_ha = [0.1080, 0.0611, 0.0420]
colors_b = [c_nano, c_af2, "#94a3b8"]
sizes_b = [450, 520, 260]

ax_b.scatter(params, gdt_ha, s=sizes_b, c=colors_b, edgecolor="#0f172a", linewidth=1.5, zorder=4)

ax_b.set_xscale("log")
ax_b.set_xlabel("Trainable Parameters (Millions, log scale)", labelpad=8)
ax_b.set_ylabel("GDT_HA (Global Backbone Accuracy)", labelpad=8)
ax_b.set_title("B   Biological Data-Efficiency Frontier", loc="left", fontweight="bold", pad=12, fontsize=13)

# Shaded budget frontier (< 3.0M parameter budget)
ax_b.axvspan(0.8, 3.2, alpha=0.12, color="#14b8a6", zorder=1)
ax_b.text(0.9, 0.122, "Competition Budget Frontier (< 3.0M)", fontsize=9, fontweight="bold", color="#0f766e", style="italic")

# Point annotations with arrows
ax_b.annotate("NanoFold v1 (Ours)\n2.83M params\n(0.1080 GDT_HA)",
              xy=(2.83, 0.1080), xytext=(4.2, 0.103),
              fontweight="bold", color="#0f766e", fontsize=10,
              arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.5, connectionstyle="arc3,rad=-0.1"))

ax_b.annotate("AF2 Full Baseline\n94.2M params (Overfitting)\n(0.0611 GDT_HA)",
              xy=(94.2, 0.0611), xytext=(22, 0.075),
              fontweight="bold", color="#334155", fontsize=9.5,
              arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5, connectionstyle="arc3,rad=0.1"))

ax_b.annotate("AF2 Tiny\n1.4M params",
              xy=(1.4, 0.0420), xytext=(1.65, 0.033),
              color="#64748b", fontsize=9)

ax_b.set_xlim(0.8, 200)
ax_b.set_ylim(0.02, 0.132)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)


# ==============================================================
# PANEL C: Full 8-Metric CASP15 Profile Delta (Bottom Wide Span)
# ==============================================================
ax_c = fig.add_subplot(gs[1, :])
ax_c.set_facecolor("#ffffff")
ax_c.grid(axis="x", linestyle="--", color="#cbd5e1", zorder=0)

all_metrics = [
    "SphereGrinder (Local Environment RMSD)",
    "All-Atom14 lDDT (Distance Preservation)",
    "GDT_HA (Global Cα Backbone Alignment)",
    "Backbone Dihedrals (Ramachandran Angles)",
    "DipDiff (Local Triplet Discrepancy)",
    "Side-Chain Geometry (Rotamer χ Angles)",
    "CADaa (Contact Area Difference)",
    "Steric Clash Score (MolProbity Overlap)",
]

# Relative percentage deltas vs AF2 Full
pct_deltas = [+334.8, +92.5, +76.8, +2.4, -24.1, -34.8, -78.9, -98.1]
abs_scores_nano = [0.3639, 0.2945, 0.1080, 0.6186, 0.5052, 0.4744, 0.0843, 0.0063]
abs_scores_af2 = [0.0837, 0.1530, 0.0611, 0.6042, 0.6652, 0.7271, 0.4009, 0.3252]
casp_weights = [0.0938, 0.0938, 0.2500, 0.1250, 0.1250, 0.0938, 0.0938, 0.1250]

y_pos = np.arange(len(all_metrics))
bar_colors = [c_nano if v > 0 else (c_accent if abs(v) > 50 else c_trailing) for v in pct_deltas]

bars = ax_c.barh(y_pos, pct_deltas, height=0.55, color=bar_colors, edgecolor="#0f172a", linewidth=1.0, zorder=3)

# Vertical zero reference
ax_c.axvline(0, color="#0f172a", linewidth=1.4, zorder=4)

ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(all_metrics, fontsize=11, fontweight="medium")
ax_c.invert_yaxis()
ax_c.set_xlabel("Relative Performance Delta vs. minAlphaFold2 Full Baseline (%)", labelpad=8, fontsize=12)
ax_c.set_title("C   Full CASP15 Benchmark Performance Profile (NanoFold v1 vs. minAlphaFold2 Full)", loc="left", fontweight="bold", pad=12, fontsize=13)
ax_c.set_xlim(-118, 380)

# Value annotations on each bar: Delta % and absolute score comparison
for i, (bar, delta_val) in enumerate(zip(bars, pct_deltas)):
    w = bar.get_width()
    offset = 6 if w >= 0 else -6
    ha = "left" if w >= 0 else "right"
    sign = "+" if delta_val > 0 else ""
    score_text = f"{sign}{delta_val:.1f}%  (NanoFold: {abs_scores_nano[i]:.4f} vs. AF2: {abs_scores_af2[i]:.4f})"
    ax_c.text(w + offset, bar.get_y() + bar.get_height()/2, score_text,
              va="center", ha=ha, fontsize=9.5, fontweight="bold", color="#0f172a")

# Clear, uncrowded callout box for the Steric Clash Bottleneck
ax_c.annotate("Steric Clash Bottleneck\nScore: 0.0063 vs. 0.3252\n(Need explicit repulsion potential)",
              xy=(-98.1, 7), xytext=(-85, 5.0),
              arrowprops=dict(facecolor=c_accent, arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=-0.15"),
              fontsize=9.5, color=c_accent, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.4", fc="#fff1f2", ec=c_accent, lw=1.2))

# Outcome legend patches in bottom right
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=c_nano, edgecolor="#0f172a", label="Superior Performance (+Δ%)"),
    Patch(facecolor=c_trailing, edgecolor="#0f172a", label="Moderate Deficit (-Δ%)"),
    Patch(facecolor=c_accent, edgecolor="#0f172a", label="Identified Bottleneck (Clash / Contact)"),
]
ax_c.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9.5)

ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

# Add overarching Figure Caption / Note at bottom
fig.text(0.06, 0.015,
         "Figure 1 | Official CASP15 validation benchmark on 1,000 PDB test chains under biological data scarcity (10,000 training chains, 30,000 steps).\n"
         "A: Tertiary and local structure discovery gains. B: Model scale vs. global fold accuracy (< 3.0M budget frontier). C: Comprehensive 8-metric CASP15 delta profile.",
         fontsize=9.5, color="#334155", style="italic")

# Save high-res figure
output_brain = "/Users/admin/.gemini/antigravity-ide/brain/cddb69d0-c987-4304-999f-fc09cd12f160/nanofold_v1_benchmark_figure.png"
output_assets = "assets/nanofold_v1_benchmark_figure.png"

fig.savefig(output_brain, dpi=300, bbox_inches="tight")
fig.savefig(output_assets, dpi=300, bbox_inches="tight")
print("Clean, uncrowded publication figure saved successfully!")
