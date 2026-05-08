import os
import pandas as pd
import matplotlib.pyplot as plt

INFILE = "results/dynamics/identity_displacement.csv"
OUTDIR = "results/dynamics"
os.makedirs(OUTDIR, exist_ok=True)

THRESHOLD = 0.2

df = pd.read_csv(INFILE)
high = df[df["displacement"] > THRESHOLD].copy()

def assign_axis(row):
    pair = {row["top1"], row["top2"]}

    if pair == {"T", "NK"}:
        return "cytotoxic_T_NK_axis"

    if pair == {"Mono", "DC"}:
        return "myeloid_DC_axis"

    if "Platelet" in pair:
        return "platelet_associated_axis"

    return "other_axis"

high["displacement_axis"] = high.apply(assign_axis, axis=1)

axis_summary = (
    high.groupby("displacement_axis")["displacement"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_cells"})
        .sort_values("n_cells", ascending=False)
)

axis_summary["fraction_of_high_displacement"] = axis_summary["n_cells"] / len(high)

axis_summary.to_csv(f"{OUTDIR}/high_displacement_axis_summary.csv", index=False)

plt.figure(figsize=(6, 4))
plt.bar(axis_summary["displacement_axis"], axis_summary["n_cells"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("High-displacement cells")
plt.xlabel("Identity-displacement axis")
plt.title(f"High-displacement cells by axis (displacement > {THRESHOLD})")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/high_displacement_axis_counts.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 4))
plt.bar(axis_summary["displacement_axis"], axis_summary["mean"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean displacement")
plt.xlabel("Identity-displacement axis")
plt.title("Mean displacement by identity axis")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/high_displacement_axis_mean.png", dpi=300)
plt.close()

print(axis_summary)
print()
print("Saved:")
print(f"{OUTDIR}/high_displacement_axis_summary.csv")
print(f"{OUTDIR}/high_displacement_axis_counts.png")
print(f"{OUTDIR}/high_displacement_axis_mean.png")
