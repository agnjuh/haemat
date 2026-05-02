import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN = "results/top5_union_heatmap.csv"
OUTDIR = "results/program_decomposition"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(IN, index_col=0)

# Normalization (per gene across cell types)
row_sum = df.sum(axis=1).replace(0, np.nan)
norm = df.div(row_sum, axis=0).fillna(0)

# Metrics
specificity = norm.max(axis=1)
dominant = norm.idxmax(axis=1)
shared_n = (norm >= 0.15).sum(axis=1)

summary = pd.DataFrame({
    "gene": df.index,
    "dominant_cell_type": dominant.values,
    "specificity": specificity.values,
    "shared_celltype_count": shared_n.values,
    "total_importance": df.sum(axis=1).values
})

# NEW classification logic (3-tier)
def label_program(row):
    if row["specificity"] >= 0.85:
        return "cell_type_specific"
    elif row["specificity"] >= 0.5:
        return "intermediate_program"
    else:
        return "shared_program"

summary["program_type"] = summary.apply(label_program, axis=1)

# Sorting
summary = summary.sort_values(
    ["program_type", "dominant_cell_type", "specificity", "total_importance"],
    ascending=[True, True, False, False]
)

# Save outputs
summary.to_csv(f"{OUTDIR}/gene_program_decomposition.csv", index=False)

summary[summary["program_type"] == "shared_program"].to_csv(
    f"{OUTDIR}/shared_program_genes.csv", index=False
)

summary[summary["program_type"] == "intermediate_program"].to_csv(
    f"{OUTDIR}/intermediate_program_genes.csv", index=False
)

summary[summary["program_type"] == "cell_type_specific"].to_csv(
    f"{OUTDIR}/cell_type_specific_genes.csv", index=False
)

# Plot: specificity distribution
plt.figure(figsize=(6,4))
plt.hist(summary["specificity"], bins=15)
plt.xlabel("Specificity score")
plt.ylabel("Number of genes")
plt.title("Distribution of gene specificity")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/gene_specificity_distribution.png", dpi=300)
plt.close()

# Plot: program composition
counts = summary["program_type"].value_counts()
plt.figure(figsize=(5,4))
plt.bar(counts.index, counts.values)
plt.ylabel("Number of genes")
plt.title("Gene program composition")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/program_type_counts.png", dpi=300)
plt.close()

# Optional: heatmap of normalized contributions
plt.figure(figsize=(8,10))
plt.imshow(norm.values, aspect='auto')
plt.yticks(range(len(norm.index)), norm.index)
plt.xticks(range(len(norm.columns)), norm.columns)
plt.colorbar(label="Normalized importance")
plt.title("Normalized gene contributions across cell types")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/normalized_heatmap.png", dpi=300)
plt.close()

print("Updated program decomposition outputs saved to:")
print(OUTDIR)
