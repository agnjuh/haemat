import os
import numpy as np
import pandas as pd
import scanpy as sc

OUTDIR = "results/program_scores"
os.makedirs(OUTDIR, exist_ok=True)

DATA = "data/pbmc10k.h5ad"

PROGRAMS = {
    "B_program": ["MS4A1", "CD79A", "BANK1", "CD79B", "CD37"],
    "T_program": ["CD3D", "CD3E", "TRAC", "LCK", "IL7R", "LTB"],
    "NK_cytotoxic_program": ["NKG7", "GNLY", "PRF1", "GZMB", "GZMH", "KLRD1", "CCL5"],
    "Myeloid_program": ["LYZ", "S100A8", "S100A9", "AIF1", "TYROBP", "FCGR3A"],
    "DC_program": ["FCER1A", "CLEC10A", "CD1C", "IRF8"],
    "Platelet_program": ["PPBP", "PF4", "NRGN", "GP9"],
}

print(f"Reading: {DATA}")
adata = sc.read_h5ad(DATA)

X = adata.raw.X if adata.raw is not None else adata.X
var_names = adata.raw.var_names if adata.raw is not None else adata.var_names

if not isinstance(var_names, pd.Index):
    var_names = pd.Index(var_names)

scores = pd.DataFrame(index=adata.obs_names)

for program_name, genes in PROGRAMS.items():
    present = [g for g in genes if g in var_names]

    if len(present) == 0:
        scores[program_name] = np.nan
        print(f"{program_name}: 0 genes found")
        continue

    idx = [var_names.get_loc(g) for g in present]
    sub = X[:, idx]

    if hasattr(sub, "toarray"):
        sub = sub.toarray()

    scores[program_name] = np.asarray(sub).mean(axis=1)
    print(f"{program_name}: {len(present)}/{len(genes)} genes found -> {present}")

scores.index.name = "cell_id"
scores.to_csv(f"{OUTDIR}/program_scores.csv")

summary = scores.describe().T
summary.to_csv(f"{OUTDIR}/program_score_summary.csv")

print("Saved:")
print(f"{OUTDIR}/program_scores.csv")
print(f"{OUTDIR}/program_score_summary.csv")
