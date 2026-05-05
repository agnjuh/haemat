import os
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "results/transitional_states"
os.makedirs(OUTDIR, exist_ok=True)

TABLE = f"{OUTDIR}/transitional_state_table.csv"
df = pd.read_csv(TABLE)

# 1. Counts of transitional categories
counts = df["transitional_state"].value_counts()

plt.figure(figsize=(7, 4))
plt.bar(counts.index, counts.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of cells")
plt.xlabel("State category")
plt.title("Transitional and ambiguous state categories")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/transitional_state_counts.png", dpi=300)
plt.close()

# 2. Entropy vs top2 gap
plt.figure(figsize=(6, 5))
for state, sub in df.groupby("transitional_state"):
    plt.scatter(sub["entropy"], sub["top2_gap"], s=6, alpha=0.5, label=state)

plt.xlabel("Entropy")
plt.ylabel("Top-2 probability gap")
plt.title("Ambiguity landscape")
plt.legend(frameon=False, fontsize=7, markerscale=2)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/ambiguity_landscape.png", dpi=300)
plt.close()

# 3. Decision vs transitional state
ct = pd.crosstab(df["decision"], df["transitional_state"])

plt.figure(figsize=(8, 5))
plt.imshow(ct.values, aspect="auto")
plt.colorbar(label="Cells")
plt.xticks(range(len(ct.columns)), ct.columns, rotation=45, ha="right")
plt.yticks(range(len(ct.index)), ct.index)
plt.xlabel("State category")
plt.ylabel("HAEMAT decision")
plt.title("HAEMAT decisions by transitional state category")

for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        plt.text(j, i, str(ct.iloc[i, j]), ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/decision_by_transitional_state.png", dpi=300)
plt.close()

print("Saved:")
print(f"{OUTDIR}/transitional_state_counts.png")
print(f"{OUTDIR}/ambiguity_landscape.png")
print(f"{OUTDIR}/decision_by_transitional_state.png")
