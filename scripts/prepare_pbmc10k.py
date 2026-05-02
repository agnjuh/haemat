import glob
import os
import scanpy as sc

BASE = "data/external_pbmc"
OUT = "data/pbmc10k.h5ad"

# Find the 10x filtered_feature_bc_matrix folder automatically
candidates = glob.glob(f"{BASE}/**/filtered_feature_bc_matrix", recursive=True)

if not candidates:
    raise FileNotFoundError(
        "Could not find a filtered_feature_bc_matrix directory under "
        f"{BASE}. Check that the 10x tar.gz file was downloaded and extracted."
    )

# Prefer a directory that actually contains matrix.mtx.gz
valid = [
    p for p in candidates
    if os.path.exists(os.path.join(p, "matrix.mtx.gz"))
]

if not valid:
    raise FileNotFoundError(
        "Found filtered_feature_bc_matrix directories, but none contain matrix.mtx.gz.\n"
        f"Candidates found: {candidates}"
    )

path = valid[0]
print(f"Reading 10x matrix from: {path}")

adata = sc.read_10x_mtx(
    path,
    var_names="gene_symbols",
    cache=True
)

adata.var_names_make_unique()

os.makedirs("data", exist_ok=True)
adata.write(OUT)

print(f"Saved: {OUT}")
print(adata)
