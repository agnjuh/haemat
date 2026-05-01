import scanpy as sc

adata = sc.read_10x_mtx(
    "data/filtered_gene_bc_matrices/hg19/",
    var_names='gene_symbols',
    cache=True
)

adata.var_names_make_unique()

# basic preprocessing (ugyanaz, mint trainben!)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata.write("data/pbmc3k.h5ad")
