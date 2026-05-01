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

OUTDIR = "results/hierarchical"
os.makedirs(OUTDIR, exist_ok=True)

RANDOM_SEED = 42

LINEAGE_MAP = {
    "lymphoid": ["B", "T", "NK"],
    "myeloid": ["Mono", "DC"],
    "platelet": ["Platelet"],
}

def _to_dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)

def class_to_lineage(y):
    for lin, members in LINEAGE_MAP.items():
        if y in members:
            return lin
    return "other"

def save_cm(y_true, y_pred, labels, out_png, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def preprocess_train_test(ad_tr, ad_te, marker_sets):
    common = ad_tr.var_names.intersection(ad_te.var_names)
    ad_tr = ad_tr[:, common].copy()
    ad_te = ad_te[:, common].copy()

    for a in (ad_tr, ad_te):
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)

    try:
        sc.pp.highly_variable_genes(ad_tr, n_top_genes=3000, flavor="seurat_v3")
    except Exception:
        sc.pp.highly_variable_genes(ad_tr, n_top_genes=3000, flavor="cell_ranger")

    marker_genes = set(g for genes in marker_sets.values() for g in genes)

    hvg_train = [
        g for g in ad_tr.var.index[ad_tr.var["highly_variable"]]
        if g in ad_te.var_names
    ]

    marker_genes_present = [
        g for g in marker_genes
        if g in ad_tr.var_names and g in ad_te.var_names
    ]

    features = sorted(set(hvg_train).union(marker_genes_present))

    ad_tr = ad_tr[:, features].copy()
    ad_te = ad_te[:, features].copy()

    Xtr = _to_dense(ad_tr.X)
    Xte = _to_dense(ad_te.X)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    n_components = min(50, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train = pca.fit_transform(Xtr_s)
    X_test = pca.transform(Xte_s)

    return X_train, X_test, ad_tr, ad_te

def fit_predict_model(X_train, y_train, X_test):
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    return clf, pred, proba

def main():
    marker_sets = load_curated_csv("markers_curated.csv", tiers=("core",))

    ad_train = sc.read_h5ad("data/5k_pbmc_10x.h5ad")
    ad_test = sc.read_h5ad("data/pbmc3k.h5ad")
    ad_train.var_names_make_unique()
    ad_test.var_names_make_unique()

    label_col_train = assign_proxy_labels(ad_train, marker_sets, out_col="cell_type")
    label_col_test = assign_proxy_labels(ad_test, marker_sets, out_col="cell_type")

    y_train_cell = ad_train.obs[label_col_train].astype(str).values
    y_test_cell = ad_test.obs[label_col_test].astype(str).values

    y_train_lineage = np.array([class_to_lineage(y) for y in y_train_cell])
    y_test_lineage = np.array([class_to_lineage(y) for y in y_test_cell])

    X_train, X_test, ad_train_pp, ad_test_pp = preprocess_train_test(ad_train, ad_test, marker_sets)

    # Level 1: lineage classifier
    lin_clf, pred_lineage, proba_lineage = fit_predict_model(X_train, y_train_lineage, X_test)
    lineage_labels = sorted(np.unique(y_train_lineage))

    lineage_report = classification_report(y_test_lineage, pred_lineage, labels=lineage_labels, digits=3)
    lineage_macro_f1 = f1_score(y_test_lineage, pred_lineage, labels=lineage_labels, average="macro")
    lineage_bal_acc = balanced_accuracy_score(y_test_lineage, pred_lineage)

    with open(f"{OUTDIR}/lineage_model_report.txt", "w") as f:
        f.write(lineage_report)
        f.write(f"\nmacro-F1={lineage_macro_f1:.3f}\n")
        f.write(f"balanced-acc={lineage_bal_acc:.3f}\n")

    save_cm(
        y_test_lineage,
        pred_lineage,
        lineage_labels,
        f"{OUTDIR}/confusion_lineage_model.png",
        "Hierarchical model level 1: lineage",
    )

    # Level 2: within-lineage models
    final_pred = []
    model_summaries = []

    for lin in y_test_lineage:
        final_pred.append("Unknown")
    final_pred = np.array(final_pred, dtype=object)

    for lin, members in LINEAGE_MAP.items():
        if lin == "platelet":
            final_pred[pred_lineage == "platelet"] = "Platelet"
            continue

        train_mask = y_train_lineage == lin
        test_mask = pred_lineage == lin

        y_train_sub = y_train_cell[train_mask]

        # only train if at least 2 classes available
        if len(np.unique(y_train_sub)) < 2 or test_mask.sum() == 0:
            continue

        sub_clf, sub_pred, sub_proba = fit_predict_model(
            X_train[train_mask],
            y_train_sub,
            X_test[test_mask],
        )

        final_pred[test_mask] = sub_pred

        y_test_sub_true = y_test_cell[test_mask]
        labels_sub = sorted(np.unique(y_train_sub))

        report_sub = classification_report(y_test_sub_true, sub_pred, labels=labels_sub, digits=3)
        macro_sub = f1_score(y_test_sub_true, sub_pred, labels=labels_sub, average="macro")

        # Balanced accuracy is computed only on the routed subset.
        # This is useful as an internal comparison, but it is not equivalent
        # to global lineage-level balanced accuracy.
        bal_sub = balanced_accuracy_score(y_test_sub_true, sub_pred)

        with open(f"{OUTDIR}/{lin}_within_lineage_report.txt", "w") as f:
            f.write(report_sub)
            f.write(f"\nmacro-F1={macro_sub:.3f}\n")
            f.write(f"balanced-acc={bal_sub:.3f}\n")

        save_cm(
            y_test_sub_true,
            sub_pred,
            labels_sub,
            f"{OUTDIR}/confusion_{lin}_within_lineage.png",
            f"Within-lineage model: {lin}",
        )

        model_summaries.append({
            "lineage": lin,
            "n_train": int(train_mask.sum()),
            "n_test_routed": int(test_mask.sum()),
            "classes": ",".join(labels_sub),
            "macro_f1": macro_sub,
            "balanced_acc": bal_sub,
        })

    cell_labels = sorted(np.unique(y_train_cell))
    final_report = classification_report(y_test_cell, final_pred, labels=cell_labels, digits=3)
    final_macro_f1 = f1_score(y_test_cell, final_pred, labels=cell_labels, average="macro")
    final_bal_acc = balanced_accuracy_score(y_test_cell, final_pred)

    with open(f"{OUTDIR}/hierarchical_celltype_report.txt", "w") as f:
        f.write(final_report)
        f.write(f"\nmacro-F1={final_macro_f1:.3f}\n")
        f.write(f"balanced-acc={final_bal_acc:.3f}\n")

    save_cm(
        y_test_cell,
        final_pred,
        cell_labels,
        f"{OUTDIR}/confusion_hierarchical_celltype.png",
        "Hierarchical model final cell type",
    )

    pd.DataFrame({
        "true": y_test_cell,
        "true_lineage": y_test_lineage,
        "pred_lineage": pred_lineage,
        "final_pred": final_pred,
    }).to_csv(f"{OUTDIR}/hierarchical_predictions.csv", index=False)

    pd.DataFrame(model_summaries).to_csv(f"{OUTDIR}/within_lineage_model_summary.csv", index=False)

    print("Hierarchical modelling complete.")
    print(f"Lineage macro-F1={lineage_macro_f1:.3f}, balanced-acc={lineage_bal_acc:.3f}")
    print(f"Final cell-type macro-F1={final_macro_f1:.3f}, balanced-acc={final_bal_acc:.3f}")

if __name__ == "__main__":
    main()
