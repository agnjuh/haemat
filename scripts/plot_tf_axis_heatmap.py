import os
import pandas as pd
import matplotlib.pyplot as plt

INFILE = "results/regulatory/tf_network/tf_axis_edges.csv"
OUTDIR = "results/regulatory/tf_network"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INFILE)

keep = {
    "Cytotoxic T/NK": ("cytotoxic_T_NK", ["RELA", "FLI1", "GATA3"]),
    "Myeloid/DC": ("myeloid_DC", ["SPI1", "NR1H3", "RUNX1"]),
    "Platelet-associated": ("platelet_associated", ["SPI1", "RUNX1", "GATA2", "LMO2", "LYL1", "CEBPB"]),
}

rows = []

for display_axis, (axis, tfs) in keep.items():
    for tf in tfs:
        sub = df[(df["axis"] == axis) & (df["TF"] == tf)]
        if sub.empty:
            score = 0
            pval = None
        else:
            score = sub["max_combined_score"].max()
            pval = sub["min_p"].min()

        rows.append({
            "axis": display_axis,
            "TF": tf,
            "combined_score": score,
            "p_value": pval
        })

plot_df = pd.DataFrame(rows)

tf_order = ["RELA", "FLI1", "GATA3", "SPI1", "NR1H3", "RUNX1", "GATA2", "LMO2", "LYL1", "CEBPB"]
axis_order = ["Cytotoxic T/NK", "Myeloid/DC", "Platelet-associated"]

mat = (
    plot_df.pivot(index="axis", columns="TF", values="combined_score")
    .reindex(index=axis_order, columns=tf_order)
    .fillna(0)
)

plt.figure(figsize=(9, 3.8))
im = plt.imshow(mat.values, aspect="auto")

plt.xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
plt.yticks(range(len(mat.index)), mat.index)

cbar = plt.colorbar(im)
cbar.set_label("Combined score")

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat.iloc[i, j]
        if val > 0:
            plt.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)

plt.title("Key TF signals across identity-displacement axes")
plt.xlabel("Transcription factor")
plt.ylabel("Identity-displacement axis")
plt.tight_layout()

plt.savefig(f"{OUTDIR}/tf_axis_heatmap.png", dpi=300)
plt.savefig(f"{OUTDIR}/tf_axis_heatmap.pdf")
plt.close()

plot_df.to_csv(f"{OUTDIR}/tf_axis_heatmap_values.csv", index=False)

print("Saved:")
print(f"{OUTDIR}/tf_axis_heatmap.png")
print(f"{OUTDIR}/tf_axis_heatmap.pdf")
print(f"{OUTDIR}/tf_axis_heatmap_values.csv")
