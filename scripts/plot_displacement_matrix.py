import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

INFILE = "results/dynamics/identity_displacement.csv"
OUT = "results/dynamics/displacement_matrix.png"

df = pd.read_csv(INFILE)

pivot = df.pivot_table(
    index="top1",
    columns="top2",
    values="displacement",
    aggfunc="mean"
)

plt.figure(figsize=(6,5))
sns.heatmap(pivot, annot=True, fmt=".2f")

plt.title("Mean identity displacement (top1 → top2)")
plt.tight_layout()
plt.savefig(OUT, dpi=300)
plt.close()

print("Saved:", OUT)
