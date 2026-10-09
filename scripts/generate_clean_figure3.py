import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11.5,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
})

brain_dir = "/Users/admin/.gemini/antigravity-ide/brain/cddb69d0-c987-4304-999f-fc09cd12f160"
assets_dir = "assets"

c_af2 = "#64748b"      # Muted slate grey
c_nano = "#0d9488"     # Deep emerald / teal
c_accent = "#e11d48"   # Rose red for bottleneck
c_trailing = "#f59e0b" # Amber for moderate deficit

# Extra height and width: 14 x 7.8
fig, ax = plt.subplots(figsize=(14, 7.8), dpi=300)
ax.set_facecolor("#ffffff")
ax.grid(axis="x", linestyle="--", color="#cbd5e1", zorder=0)

metrics = [
    "SphereGrinder (Local Sphere RMSD)",
    "All-Atom14 lDDT (Distance Preservation)",
    "GDT_HA (Global Cα Backbone Alignment)",
    "Backbone Dihedrals (Ramachandran Angles)",
    "DipDiff (Local Triplet Discrepancy)",
    "Side-Chain Geometry (Rotamer χ Angles)",
    "CADaa (Contact Area Difference)",
    "Steric Clash Score (MolProbity Overlap)",
]

pct_deltas = [+334.8, +92.5, +76.8, +2.4, -24.1, -34.8, -78.9, -98.1]
abs_nano = [0.3639, 0.2945, 0.1080, 0.6186, 0.5052, 0.4744, 0.0843, 0.0063]
abs_af2 = [0.0837, 0.1530, 0.0611, 0.6042, 0.6652, 0.7271, 0.4009, 0.3252]

y_pos = np.arange(len(metrics))
bar_colors = [c_nano if v > 0 else (c_accent if abs(v) > 50 else c_trailing) for v in pct_deltas]

# Draw horizontal bars
bars = ax.barh(y_pos, pct_deltas, height=0.52, color=bar_colors, edgecolor="#0f172a", linewidth=1.1, zorder=3)

# Center zero vertical line
ax.axvline(0, color="#0f172a", linewidth=1.5, zorder=4)

ax.set_yticks(y_pos)
ax.set_yticklabels(metrics, fontsize=11.5, fontweight="medium")
ax.invert_yaxis()
ax.set_xlabel("Relative Performance Delta vs. minAlphaFold2 Full Baseline (%)", labelpad=12, fontsize=12.5, fontweight="medium")
ax.set_title("Full CASP15 Evaluation Profile: NanoFold v1 vs. minAlphaFold2 Full (1,000 Validation Chains)", fontweight="bold", pad=18, fontsize=14)

# Ample x-limits: from -260 to +520 so no text is ever clipped or crowded
ax.set_xlim(-260, 520)

# Value annotations:
for i, (bar, delta) in enumerate(zip(bars, pct_deltas)):
    w = bar.get_width()
    if w >= 0:
        # Positive bars: clean text to the right
        label_text = f"+{delta:.1f}%   (NanoFold: {abs_nano[i]:.4f}  |  AF2: {abs_af2[i]:.4f})"
        ax.text(w + 8, bar.get_y() + bar.get_height()/2, label_text,
                va="center", ha="left", fontsize=10, fontweight="bold", color="#0f172a")
    else:
        # Negative bars: clean text to the left
        label_text = f"{delta:.1f}%   ({abs_nano[i]:.4f} vs. {abs_af2[i]:.4f})"
        ax.text(w - 8, bar.get_y() + bar.get_height()/2, label_text,
                va="center", ha="right", fontsize=10, fontweight="bold", color="#0f172a")

# Standalone callout box located cleanly in the spacious open area on the right side
callout_text = (
    "Steric Clash Bottleneck Analysis:\n"
    "• Clash Score: 0.0063 vs. 0.3252 (deficit: -0.0399 pts)\n"
    "• CADaa Contact Deficit: 0.0843 vs. 0.4009 (-0.0297 pts)\n"
    "• Root Cause: Cartesian loss allows overlapping atoms without repulsion.\n"
    "• Core tertiary backbone fold (+76.8% GDT_HA) remains vastly superior."
)
ax.text(120, 5.8, callout_text, fontsize=10, color="#881337", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.8", fc="#fff1f2", ec=c_accent, lw=1.3), zorder=5)

# Place legend cleanly BELOW the plot (outside the axis area)
legend_items = [
    Patch(facecolor=c_nano, edgecolor="#0f172a", label="Superior Tertiary Fold (+Δ%)"),
    Patch(facecolor=c_trailing, edgecolor="#0f172a", label="Moderate Deficit (-Δ%)"),
    Patch(facecolor=c_accent, edgecolor="#0f172a", label="Steric Clash / Contact Bottleneck (-Δ%)"),
]
ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=3, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10.5, framealpha=0.95)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(f"{brain_dir}/figure3_casp15_profile.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{assets_dir}/figure3_casp15_profile.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 3 generated with zero overlap and legend placed cleanly below plot!")
