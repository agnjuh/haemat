import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

OUTDIR = "results/benchmark_plots"
os.makedirs(OUTDIR, exist_ok=True)

haemat = pd.read_csv("results/cross_dataset/decision_table.csv")
ct = pd.read_csv("results/celltypist_pbmc10k_mapped.csv")

df = haemat.copy()
df["celltypist"] = ct["mapped"].values

labels = ["B", "DC", "Mono", "NK", "Platelet", "T"]

# 1. True vs haemat decision
cm_haemat = confusion_matrix(df["true"], df["decision"], labels=labels)
plt.figure(figsize=(7, 6))
plt.imshow(cm_haemat, aspect="auto")
plt.colorbar(label="Count")
plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.yticks(range(len(labels)), labels)
plt.xlabel("haemat decision")
plt.ylabel("Proxy truth")
plt.title("haemat model vs proxy labels")
for i in range(cm_haemat.shape[0]):
    for j in range(cm_haemat.shape[1]):
        plt.text(j, i, str(cm_haemat[i, j]), ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/confusion_haemat_vs_proxy.png", dpi=300)
plt.close()

# 2. True vs CellTypist
cm_ct = confusion_matrix(df["true"], df["celltypist"], labels=labels)
plt.figure(figsize=(7, 6))
plt.imshow(cm_ct, aspect="auto")
plt.colorbar(label="Count")
plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.yticks(range(len(labels)), labels)
plt.xlabel("CellTypist")
plt.ylabel("Proxy truth")
plt.title("CellTypist vs proxy labels")
for i in range(cm_ct.shape[0]):
    for j in range(cm_ct.shape[1]):
        plt.text(j, i, str(cm_ct[i, j]), ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/confusion_celltypist_vs_proxy.png", dpi=300)
plt.close()

# 3. haemat decision vs CellTypist agreement
agree = df["decision"] == df["celltypist"]
agreement_summary = (
    df.assign(agree=agree)
      .groupby("true")["agree"]
      .agg(["mean", "count"])
      .reset_index()
      .rename(columns={"mean": "agreement_rate", "count": "n"})
)
agreement_summary.to_csv(f"{OUTDIR}/agreement_by_proxy_class.csv", index=False)

plt.figure(figsize=(6, 4))
plt.bar(agreement_summary["true"], agreement_summary["agreement_rate"])
plt.ylim(0, 1)
plt.ylabel("Agreement rate")
plt.xlabel("Proxy class")
plt.title("haemat model vs CellTypist agreement")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/agreement_by_proxy_class.png", dpi=300)
plt.close()

# 4. NK diagnostic: where true NK cells go
nk = df[df["true"] == "NK"]

nk_haemat = nk["decision"].value_counts().reindex(labels + ["Unknown"], fill_value=0)
nk_ct = nk["celltypist"].value_counts().reindex(labels, fill_value=0)

nk_summary = pd.DataFrame({
    "haemat": nk_haemat,
    "celltypist": nk_ct.reindex(nk_haemat.index, fill_value=0)
})
nk_summary.to_csv(f"{OUTDIR}/nk_failure_distribution.csv")

x = range(len(nk_summary.index))
width = 0.4

plt.figure(figsize=(8, 4))
plt.bar([i - width/2 for i in x], nk_summary["haemat"], width=width, label="haemat model")
plt.bar([i + width/2 for i in x], nk_summary["celltypist"], width=width, label="CellTypist")
plt.xticks(list(x), nk_summary.index, rotation=45, ha="right")
plt.ylabel("Number of true NK cells")
plt.title("True NK cells: predicted/assigned labels")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/nk_failure_comparison.png", dpi=300)
plt.close()

# --- 5. Uncertainty distributions by correctness ---
df["correct"] = df["decision"] == df["true"]

plt.figure(figsize=(6, 4))
plt.hist(df.loc[df["correct"], "entropy"], bins=30, alpha=0.7, label="Correct")
plt.hist(df.loc[~df["correct"], "entropy"], bins=30, alpha=0.7, label="Incorrect")
plt.xlabel("Entropy")
plt.ylabel("Cells")
plt.title("Entropy: correct vs incorrect decisions")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/entropy_correct_vs_incorrect.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(df.loc[df["correct"], "top2_gap"], bins=30, alpha=0.7, label="Correct")
plt.hist(df.loc[~df["correct"], "top2_gap"], bins=30, alpha=0.7, label="Incorrect")
plt.xlabel("Top-2 probability gap")
plt.ylabel("Cells")
plt.title("Top-2 gap: correct vs incorrect decisions")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/top2_gap_correct_vs_incorrect.png", dpi=300)
plt.close()

print("Saved benchmark plots to:", OUTDIR)
print("Saved:")
print(f"{OUTDIR}/confusion_haemat_vs_proxy.png")
print(f"{OUTDIR}/confusion_celltypist_vs_proxy.png")
print(f"{OUTDIR}/agreement_by_proxy_class.png")
print(f"{OUTDIR}/nk_failure_comparison.png")
print(f"{OUTDIR}/entropy_correct_vs_incorrect.png")
print(f"{OUTDIR}/top2_gap_correct_vs_incorrect.png")
