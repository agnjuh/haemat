import os
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "results/transitional_states"
os.makedirs(OUTDIR, exist_ok=True)

TABLE = f"{OUTDIR}/transitional_state_table.csv"
df = pd.read_csv(TABLE)

program_cols = [
    "B_program",
    "T_program",
    "NK_cytotoxic_program",
    "Myeloid_program",
    "DC_program",
    "Platelet_program",
]

missing = [c for c in program_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing program score columns: {missing}")

state_order = [
    "confident",
    "confident_cytotoxic_T_like",
    "cytotoxic_transitional",
    "myeloid_DC_transitional",
    "platelet_like_ambiguous",
    "high_uncertainty",
    "unknown_low_confidence",
]

state_order = [s for s in state_order if s in df["transitional_state"].unique()]

mean_scores = (
    df.groupby("transitional_state")[program_cols]
      .mean()
      .reindex(state_order)
)

mean_scores.to_csv(f"{OUTDIR}/mean_program_scores_by_state.csv")

plt.figure(figsize=(8, 4.8))
plt.imshow(mean_scores.values, aspect="auto")
plt.colorbar(label="Mean program score")

plt.xticks(
    range(len(mean_scores.columns)),
    [c.replace("_program", "").replace("_", " ") for c in mean_scores.columns],
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(mean_scores.index)),
    mean_scores.index
)

plt.xlabel("Gene program")
plt.ylabel("State category")
plt.title("Mean lineage program scores by transitional state")

for i in range(mean_scores.shape[0]):
    for j in range(mean_scores.shape[1]):
        value = mean_scores.iloc[i, j]
        plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/program_scores_by_transitional_state.png", dpi=300)
plt.close()

# Also generate a version excluding the dominant confident class
focused = mean_scores.drop(index="confident", errors="ignore")

plt.figure(figsize=(8, 4.2))
plt.imshow(focused.values, aspect="auto")
plt.colorbar(label="Mean program score")

plt.xticks(
    range(len(focused.columns)),
    [c.replace("_program", "").replace("_", " ") for c in focused.columns],
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(focused.index)),
    focused.index
)

plt.xlabel("Gene program")
plt.ylabel("State category")
plt.title("Program scores in ambiguous and transitional states")

for i in range(focused.shape[0]):
    for j in range(focused.shape[1]):
        value = focused.iloc[i, j]
        plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/program_scores_by_transitional_state_focused.png", dpi=300)
plt.close()

print("Saved:")
print(f"{OUTDIR}/mean_program_scores_by_state.csv")
print(f"{OUTDIR}/program_scores_by_transitional_state.png")
print(f"{OUTDIR}/program_scores_by_transitional_state_focused.png")
