import os
import scanpy as sc
import celltypist
import pandas as pd

os.makedirs("results", exist_ok=True)

adata = sc.read_h5ad("data/pbmc10k.h5ad")
adata.var_names_make_unique()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

pred = celltypist.annotate(
    adata,
    model="Immune_All_Low.pkl",
    majority_voting=True
)

labels = pred.predicted_labels.copy()

print("CellTypist output columns:")
print(labels.columns)

if "majority_voting" in labels.columns:
    final_labels = labels["majority_voting"]
elif "predicted_labels" in labels.columns:
    final_labels = labels["predicted_labels"]
else:
    final_labels = labels.iloc[:, 0]

out = pd.DataFrame({
    "celltypist_label": final_labels.astype(str).values
}, index=adata.obs_names)

out.to_csv("results/celltypist_pbmc10k_labels.csv")

print("Saved: results/celltypist_pbmc10k_labels.csv")
print(out["celltypist_label"].value_counts().head(20))
