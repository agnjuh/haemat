import os
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "results/transitional_states"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

table = pd.read_csv("results/transitional_states/transitional_state_table.csv")
summary = pd.read_csv("results/transitional_states/transitional_summary.csv")
mean_scores = pd.read_csv("results/transitional_states/mean_program_scores_by_state.csv", index_col=0)
sample_features = pd.read_csv("results/sample_level/sample_features.csv")

state_order = [
    "confident",
    "confident_cytotoxic_T_like",
    "cytotoxic_transitional",
    "myeloid_DC_transitional",
    "platelet_like_ambiguous",
    "high_uncertainty",
    "unknown_low_confidence",
]

summary["fraction"] = summary["n_cells"] / summary["n_cells"].sum()
summary = summary.set_index("transitional_state").reindex(
    [s for s in state_order if s in summary["transitional_state"].values]
)

mean_scores = mean_scores.reindex(
    [s for s in state_order if s in mean_scores.index]
)

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
axA, axB, axC, axD = axes.flatten()

def panel_label(ax, label, title):
    ax.text(
        -0.12, 1.08,
        f"{label}. {title}",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

# A. State fractions
panel_label(axA, "A", "State category fractions")
axA.bar(summary.index, summary["fraction"])
axA.set_ylabel("Fraction of cells")
axA.set_xlabel("State category")
axA.set_ylim(0, 1)
axA.tick_params(axis="x", rotation=45)
for tick in axA.get_xticklabels():
    tick.set_ha("right")

# B. Program score heatmap
panel_label(axB, "B", "Lineage program scores")
im = axB.imshow(mean_scores.values, aspect="auto")
axB.set_xticks(range(len(mean_scores.columns)))
axB.set_xticklabels(
    [c.replace("_program", "").replace("_", " ") for c in mean_scores.columns],
    rotation=45,
    ha="right"
)
axB.set_yticks(range(len(mean_scores.index)))
axB.set_yticklabels(mean_scores.index)
axB.set_xlabel("Gene program")
axB.set_ylabel("State category")

for i in range(mean_scores.shape[0]):
    for j in range(mean_scores.shape[1]):
        axB.text(j, i, f"{mean_scores.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6)

cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.03)
cbar.set_label("Mean score")

# C. Entropy vs top2 gap
panel_label(axC, "C", "Ambiguity landscape")
for state, sub in table.groupby("transitional_state"):
    axC.scatter(sub["entropy"], sub["top2_gap"], s=5, alpha=0.45, label=state)

axC.set_xlabel("Entropy")
axC.set_ylabel("Top-2 probability gap")
axC.set_xlim(0, max(0.6, table["entropy"].quantile(0.995)))
axC.legend(frameon=False, fontsize=6, markerscale=2)

# D. Sample-level feature summary
panel_label(axD, "D", "Sample-level ambiguity profile")

feature_cols = [
    "frac_confident",
    "frac_confident_cytotoxic_T_like",
    "frac_cytotoxic_transitional",
    "frac_myeloid_DC_transitional",
    "frac_platelet_like_ambiguous",
    "frac_high_uncertainty",
    "frac_unknown_low_confidence",
]

available = [c for c in feature_cols if c in sample_features.columns]
vals = sample_features.loc[0, available]

labels = [c.replace("frac_", "") for c in available]
axD.bar(labels, vals.values)
axD.set_ylabel("Fraction")
axD.set_xlabel("Sample-level feature")
axD.set_ylim(0, 1)
axD.tick_params(axis="x", rotation=45)
for tick in axD.get_xticklabels():
    tick.set_ha("right")

plt.tight_layout()
fig.savefig(f"{OUTDIR}/transitional_summary_figure.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUTDIR}/transitional_summary_figure.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(f"{OUTDIR}/transitional_summary_figure.png")
print(f"{OUTDIR}/transitional_summary_figure.pdf")
