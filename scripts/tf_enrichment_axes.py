import os
import pandas as pd
import gseapy as gp

GENESET_FILE = "config/axis_gene_sets.tsv"
OUTDIR = "results/regulatory/tf_enrichment"
os.makedirs(OUTDIR, exist_ok=True)

LIBRARIES = [
    "ChEA_2016",
    "ENCODE_TF_ChIP-seq_2015",
]

axis_genes = pd.read_csv(GENESET_FILE, sep="\t")

all_results = []

for axis in axis_genes["axis"].unique():
    genes = axis_genes.loc[axis_genes["axis"] == axis, "gene"].dropna().unique().tolist()

    print(f"\nRunning TF enrichment for {axis}")
    print("Genes:", genes)

    for library in LIBRARIES:
        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=library,
                organism="human",
                outdir=None,
                cutoff=1.0,
            )

            if enr.results is None or enr.results.empty:
                print(f"No results for {axis} / {library}")
                continue

            res = enr.results.copy()
            res["axis"] = axis
            res["library"] = library

            out = f"{OUTDIR}/{axis}_{library}.csv"
            res.to_csv(out, index=False)

            all_results.append(res)

            print(f"Saved: {out}")

        except Exception as e:
            print(f"FAILED: {axis} / {library}")
            print(e)

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(f"{OUTDIR}/tf_enrichment_all_axes.csv", index=False)

    top = (
        combined.sort_values(["axis", "Adjusted P-value"])
        .groupby(["axis", "library"])
        .head(10)
    )
    top.to_csv(f"{OUTDIR}/tf_enrichment_top10.csv", index=False)

    print("\nSaved combined results:")
    print(f"{OUTDIR}/tf_enrichment_all_axes.csv")
    print(f"{OUTDIR}/tf_enrichment_top10.csv")

    print("\nTop results:")
    print(top[["axis", "library", "Term", "Adjusted P-value", "Combined Score"]])
else:
    print("No enrichment results were returned.")
