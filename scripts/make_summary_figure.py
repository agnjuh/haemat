import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

OUTDIR = "results/summary_figure"
os.makedirs(OUTDIR, exist_ok=True)

MODEL_NAME = "HAEMAT"
LABELS = ["B", "DC", "Mono", "NK", "Platelet", "T"]

plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.titlesize": 11,
})

decision = pd.read_csv("results/cross_dataset/decision_table.csv")
celltypist = pd.read_csv("results/celltypist_pbmc10k_mapped.csv")

df = decision.copy()
df["celltypist"] = celltypist["mapped"].values

cm_haemat = confusion_matrix(df["true"], df["decision"], labels=LABELS)
cm_ct = confusion_matrix(df["true"], df["celltypist"], labels=LABELS)
vmax = max(cm_haemat.max(), cm_ct.max())

agreement = (
    df.assign(agree=df["decision"] == df["celltypist"])
      .groupby("true")["agree"]
      .mean()
      .reindex(LABELS)
)

nk = df[df["true"] == "NK"]
nk_haemat = nk["decision"].value_counts().reindex(LABELS + ["Unknown"], fill_value=0)
nk_ct = nk["celltypist"].value_counts().reindex(LABELS + ["Unknown"], fill_value=0)

importance = pd.read_csv("results/top5_union_heatmap.csv", index_col=0)
importance = importance.head(12)

fig = plt.figure(figsize=(11.5, 12.2), constrained_layout=False)

gs = fig.add_gridspec(
    3, 2,
    height_ratios=[1.05, 0.9, 1.15],
    hspace=0.35,
    wspace=0.25
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
axE = fig.add_subplot(gs[2, 0])
axF = fig.add_subplot(gs[2, 1])


def add_panel_label(ax, letter, title):
    ax.text(
        -0.10, 1.07,
        f"{letter}. {title}",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left"
    )


def draw_confusion(ax, cm, xlabel):
    im = ax.imshow(cm, vmin=0, vmax=vmax, aspect="equal")

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Proxy label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=5.5
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Count", fontsize=7)


add_panel_label(axA, "A", f"{MODEL_NAME} vs proxy labels")
draw_confusion(axA, cm_haemat, f"{MODEL_NAME} prediction")

add_panel_label(axB, "B", "CellTypist vs proxy labels")
draw_confusion(axB, cm_ct, "CellTypist prediction")

add_panel_label(axC, "C", f"{MODEL_NAME}–CellTypist agreement")
axC.bar(agreement.index, agreement.values, width=0.65)
axC.axhline(0.9, linestyle="--", linewidth=1)
axC.set_ylim(0, 1.05)
axC.set_ylabel("Agreement rate")
axC.set_xlabel("Proxy label")
axC.set_title("Agreement by proxy label", pad=4)

add_panel_label(axD, "D", "NK misclassification patterns")
x = np.arange(len(nk_haemat.index))
width = 0.38
axD.bar(x - width / 2, nk_haemat.values, width=width, label=MODEL_NAME)
axD.bar(x + width / 2, nk_ct.values, width=width, label="CellTypist")
axD.set_xticks(x)
axD.set_xticklabels(nk_haemat.index, rotation=45, ha="right")
axD.set_ylim(0, 700)
axD.margins(x=0.05)
axD.set_ylabel("True NK cells")
axD.set_xlabel("Assigned label")
axD.set_title("Distribution of true NK cells", pad=4)
axD.legend(frameon=False)

add_panel_label(axE, "E", "Entropy: correct vs incorrect")
df["correct"] = df["decision"] == df["true"]

bins = np.linspace(0, 0.6, 40)
axE.hist(
    df.loc[df["correct"], "entropy"],
    bins=bins,
    alpha=0.6,
    label="Correct"
)
axE.hist(
    df.loc[~df["correct"], "entropy"],
    bins=bins,
    alpha=0.6,
    label="Incorrect"
)
axE.set_yscale("log")
axE.set_xlim(0, 0.6)
axE.set_ylim(1, 2e4)
axE.set_xlabel("Entropy")
axE.set_ylabel("Cells, log scale")
axE.set_title("Prediction entropy under domain shift", pad=4)
axE.legend(frameon=False)

add_panel_label(axF, "F", "Top gene importance per cell type")
imF = axF.imshow(importance.values, aspect="auto")
axF.set_xticks(range(len(importance.columns)))
axF.set_xticklabels(importance.columns, rotation=45, ha="right")
axF.set_yticks(range(len(importance.index)))
axF.set_yticklabels(importance.index, fontsize=6)
axF.set_xlabel("Cell type")
axF.set_title("L1 importance coefficients", pad=4)

cbarF = fig.colorbar(imF, ax=axF, fraction=0.035, pad=0.02)
cbarF.ax.tick_params(labelsize=6)
cbarF.set_label("Importance", fontsize=7)

fig.subplots_adjust(top=0.96, bottom=0.05)

fig.savefig(
    f"{OUTDIR}/HAEMAT_summary_figure.png",
    dpi=300,
    bbox_inches="tight"
)
fig.savefig(
    f"{OUTDIR}/HAEMAT_summary_figure.pdf",
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:")
print(f"{OUTDIR}/HAEMAT_summary_figure.png")
print(f"{OUTDIR}/HAEMAT_summary_figure.pdf")
