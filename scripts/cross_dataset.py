import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score

from pbmc_utils import load_curated_csv, assign_proxy_labels

RANDOM_SEED = 42
os.makedirs("results/cross_dataset", exist_ok=True)


def _to_dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


# load
ad_tr = sc.read_h5ad("data/5k_pbmc_10x.h5ad")
ad_te = sc.read_h5ad("data/pbmc3k.h5ad")

ad_tr.var_names_make_unique()
ad_te.var_names_make_unique()

print("Train:", ad_tr)
print("External test:", ad_te)

# labels
marker_sets = load_curated_csv("markers_curated.csv", tiers=("core",))
label_col_tr = assign_proxy_labels(ad_tr, marker_sets, out_col="cell_type")
label_col_te = assign_proxy_labels(ad_te, marker_sets, out_col="cell_type")

# gene intersection
common = ad_tr.var_names.intersection(ad_te.var_names)
ad_tr = ad_tr[:, common].copy()
ad_te = ad_te[:, common].copy()

print(f"Common genes: {len(common)}")

# preprocessing separately, but feature selection on train only
for a in (ad_tr, ad_te):
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

try:
    sc.pp.highly_variable_genes(ad_tr, n_top_genes=3000, flavor="seurat_v3")
except Exception:
    sc.pp.highly_variable_genes(ad_tr, n_top_genes=3000, flavor="cell_ranger")

# Keep train-derived HVGs, but force curated marker genes into the feature space.
# This integrates biological prior knowledge without fitting feature selection on the test set.
marker_genes = set(g for genes in marker_sets.values() for g in genes)

hvg_train = [
    g for g in ad_tr.var.index[ad_tr.var["highly_variable"]]
    if g in ad_te.var_names
]

marker_genes_present = [
    g for g in marker_genes
    if g in ad_tr.var_names and g in ad_te.var_names
]

hvg = sorted(set(hvg_train).union(marker_genes_present))

ad_tr = ad_tr[:, hvg].copy()
ad_te = ad_te[:, hvg].copy()

pd.Series(hvg).to_csv("results/cross_dataset/hvg_genes_cross_dataset.csv", index=False)

# matrix
Xtr = _to_dense(ad_tr.X)
Xte = _to_dense(ad_te.X)

scaler = StandardScaler(with_mean=True, with_std=True)
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

pca = PCA(n_components=50, random_state=RANDOM_SEED)
X_train = pca.fit_transform(Xtr_s)
X_test = pca.transform(Xte_s)

y_train = ad_tr.obs[label_col_tr].astype(str).values
y_test = ad_te.obs[label_col_te].astype(str).values

# ensure same classes
classes = sorted(np.unique(y_train).tolist())

clf = LogisticRegression(
    max_iter=1000,
    multi_class="multinomial",
    random_state=RANDOM_SEED
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# metrics
report = classification_report(y_test, y_pred, labels=classes, digits=3)
macro_f1 = f1_score(y_test, y_pred, labels=classes, average="macro")
bal_acc = balanced_accuracy_score(y_test, y_pred)

with open("results/cross_dataset/report_cross_logreg.txt", "w") as f:
    f.write(report)
    f.write(f"\nmacro-F1={macro_f1:.3f}\n")
    f.write(f"balanced-acc={bal_acc:.3f}\n")

pd.DataFrame({
    "metric": ["macro_f1", "balanced_accuracy"],
    "value": [macro_f1, bal_acc]
}).to_csv("results/cross_dataset/metrics_cross_logreg.csv", index=False)

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=classes)

fig, ax = plt.subplots(figsize=(7.2, 5.8))
im = ax.imshow(cm, aspect="auto")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Count")

ax.set_title("Cross-dataset Logistic Regression")
ax.set_xlabel("Predicted")
ax.set_ylabel("Proxy label")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

fig.tight_layout()
fig.savefig("results/cross_dataset/confusion_cross_logreg.png", dpi=300)
plt.close(fig)

print("DONE cross-dataset evaluation.")
print(f"macro-F1={macro_f1:.3f}, balanced-acc={bal_acc:.3f}")

# SAVE for threshold analysis
np.save("results/cross_dataset/proba.npy", clf.predict_proba(X_test))
np.save("results/cross_dataset/y_test.npy", y_test)
np.save("results/cross_dataset/classes.npy", clf.classes_)

print("Saved cross-dataset proba/y_test/classes")
