import os, json
import pandas as pd
import scanpy as sc

def load_curated_csv(path="markers_curated.csv", tiers=("core", "extended")):
    assert os.path.exists(path), f"Curated marker file not found: {path}"
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    req = {"cell_type", "gene", "tier", "polarity"}
    missing = req - set(df.columns)
    assert not missing, f"Curated CSV missing columns: {sorted(missing)}"

    df = df[df["tier"].isin(tiers)].copy()
    df["polarity"] = df["polarity"].fillna("positive").str.lower()
    df = df[df["polarity"] == "positive"].copy()
    df["gene"] = df["gene"].astype(str).str.upper().str.strip()

    return {
        ct: sorted(df.loc[df["cell_type"] == ct, "gene"].dropna().unique().tolist())
        for ct in df["cell_type"].unique()
    }


def assign_proxy_labels(adata, marker_sets, out_col="cell_type"):
    label_col = None
    for c in adata.obs.columns:
        cl = c.lower()
        if ("cell" in cl and "type" in cl) or ("annotation" in cl):
            label_col = c
            break

    if label_col is not None:
        adata.obs[out_col] = adata.obs[label_col].astype(str)
        return out_col

    print("No cell-type labels found. Building proxy labels via Leiden + marker scoring...")

    tmp = adata.copy()
    tmp.var_names_make_unique()

    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)

    try:
        sc.pp.highly_variable_genes(tmp, n_top_genes=3000, flavor="seurat_v3")
        tmp_hvg = tmp[:, tmp.var.highly_variable].copy()
        sc.pp.scale(tmp_hvg, max_value=10)
        sc.tl.pca(tmp_hvg, n_comps=50, svd_solver="arpack")
        sc.pp.neighbors(tmp_hvg, n_neighbors=15, n_pcs=30)
        sc.tl.leiden(tmp_hvg, resolution=0.6, key_added="leiden")

        for ct, genes in marker_sets.items():
            present = [g for g in genes if g in tmp_hvg.var_names]
            if present:
                sc.tl.score_genes(tmp_hvg, present, score_name=f"score_{ct}", use_raw=False)
            else:
                tmp_hvg.obs[f"score_{ct}"] = 0.0

        cluster2ct = {}
        for clab in sorted(tmp_hvg.obs["leiden"].unique(), key=lambda x: int(x)):
            sub = tmp_hvg[tmp_hvg.obs["leiden"] == clab]
            means = {ct: float(sub.obs[f"score_{ct}"].mean()) for ct in marker_sets}
            cluster2ct[clab] = max(means, key=means.get)

        adata.obs[out_col] = pd.Series(
            [cluster2ct[cl_] for cl_ in tmp_hvg.obs["leiden"]],
            index=tmp_hvg.obs.index,
            dtype="object"
        )

        unlabeled_mask = adata.obs[out_col].isna()
        if unlabeled_mask.any():
            for ct, genes in marker_sets.items():
                present = [g for g in genes if g in tmp.var_names]
                if present:
                    sc.tl.score_genes(tmp, present, score_name=f"score_{ct}", use_raw=False)
                else:
                    tmp.obs[f"score_{ct}"] = 0.0

            scores = tmp.obs[[f"score_{ct}" for ct in marker_sets]].to_numpy()
            best_idx = scores.argmax(axis=1)
            labels = list(marker_sets.keys())
            proxy = pd.Series([labels[i] for i in best_idx], index=tmp.obs.index, dtype="object")
            adata.obs.loc[unlabeled_mask, out_col] = proxy.loc[unlabeled_mask].values

    except Exception as e:
        print(f"Leiden unavailable or failed ({e}). Falling back to marker-only argmax.")

        for ct, genes in marker_sets.items():
            present = [g for g in genes if g in tmp.var_names]
            if present:
                sc.tl.score_genes(tmp, present, score_name=f"score_{ct}", use_raw=False)
            else:
                tmp.obs[f"score_{ct}"] = 0.0

        scores = tmp.obs[[f"score_{ct}" for ct in marker_sets]].to_numpy()
        best_idx = scores.argmax(axis=1)
        labels = list(marker_sets.keys())
        adata.obs[out_col] = [labels[i] for i in best_idx]

    return out_col
