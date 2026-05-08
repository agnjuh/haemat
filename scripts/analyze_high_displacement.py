import os
import pandas as pd
import matplotlib.pyplot as plt

INFILE = "results/dynamics/identity_displacement.csv"
OUTDIR = "results/dynamics"
os.makedirs(OUTDIR, exist_ok=True)

THRESHOLD = 0.2

df = pd.read_csv(INFILE)

high = df[df["displacement"] > THRESHOLD].copy()

# counts by top1 -> top2 pair
pair_counts = (
    high.groupby(["top1", "top2"])
        .size()
        .reset_index(name="n_cells")
        .sort_values("n_cells", ascending=False)
)

# mean displacement by pair
pair_summary = (
    high.groupby(["top1", "top2"])["displacement"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_cells"})
        .sort_values(["n_cells", "mean"], ascending=False)
)

pair_counts.to_csv(f"{OUTDIR}/high_displacement_pair_counts.csv", index=False)
pair_summary.to_csv(f"{OUTDIR}/high_displacement_pair_summary.csv", index=False)

# matrix: counts
count_matrix = pd.crosstab(high["top1"], high["top2"])
count_matrix.to_csv(f"{OUTDIR}/high_displacement_count_matrix.csv")

# matrix: mean displacement
mean_matrix = high.pivot_table(
    index="top1",
    columns="top2",
    values="displacement",
    aggfunc="mean"
)
mean_matrix.to_csv(f"{OUTDIR}/high_displacement_mean_matrix.csv")

# plot count matrix
plt.figure(figsize=(6, 5))
plt.imshow(count_matrix.values, aspect="auto")
plt.colorbar(label="Cells")

plt.xticks(range(len(count_matrix.columns)), count_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(count_matrix.index)), count_matrix.index)

plt.xlabel("Secondary identity")
plt.ylabel("Primary identity")
plt.title(f"High identity displacement pairs (displacement > {THRESHOLD})")

for i in range(count_matrix.shape[0]):
    for j in range(count_matrix.shape[1]):
        plt.text(j, i, str(count_matrix.iloc[i, j]), ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/high_displacement_count_matrix.png", dpi=300)
plt.close()

# plot mean displacement matrix
plt.figure(figsize=(6, 5))
plt.imshow(mean_matrix.values, aspect="auto")
plt.colorbar(label="Mean displacement")

plt.xticks(range(len(mean_matrix.columns)), mean_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(mean_matrix.index)), mean_matrix.index)

plt.xlabel("Secondary identity")
plt.ylabel("Primary identity")
plt.title(f"Mean displacement among high-displacement cells")

for i in range(mean_matrix.shape[0]):
    for j in range(mean_matrix.shape[1]):
        value = mean_matrix.iloc[i, j]
        if pd.notna(value):
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/high_displacement_mean_matrix.png", dpi=300)
plt.close()

print("High displacement cells:", len(high))
print()
print(pair_summary)
print()
print("Saved:")
print(f"{OUTDIR}/high_displacement_pair_counts.csv")
print(f"{OUTDIR}/high_displacement_pair_summary.csv")
print(f"{OUTDIR}/high_displacement_count_matrix.csv")
print(f"{OUTDIR}/high_displacement_mean_matrix.csv")
print(f"{OUTDIR}/high_displacement_count_matrix.png")
print(f"{OUTDIR}/high_displacement_mean_matrix.png")
