import os
import numpy as np
import pandas as pd

OUTDIR = "results/cross_dataset"

LINEAGE_MAP = {
    "lymphoid": ["T", "NK", "B"],
    "myeloid": ["Mono", "DC"],
    "platelet": ["Platelet"],
}

PRECISION_TARGET = 0.95
THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.01), 4)


def main():
    proba = np.load(os.path.join(OUTDIR, "proba.npy"))
    y_true = np.load(os.path.join(OUTDIR, "y_test.npy"), allow_pickle=True)
    classes = np.load(os.path.join(OUTDIR, "classes.npy"), allow_pickle=True)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    true_lineage = []
    for y in y_true:
        found = "other"
        for lin, members in LINEAGE_MAP.items():
            if y in members:
                found = lin
                break
        true_lineage.append(found)
    true_lineage = np.array(true_lineage)

    lineage_scores = {}
    for lin, members in LINEAGE_MAP.items():
        idxs = [class_to_idx[c] for c in members if c in class_to_idx]
        lineage_scores[lin] = proba[:, idxs].sum(axis=1)

    lineage_df = pd.DataFrame(lineage_scores)
    pred_lineage = lineage_df.idxmax(axis=1).to_numpy()
    max_lineage_proba = lineage_df.max(axis=1).to_numpy()

    rows = []
    for lin in LINEAGE_MAP:
        is_pred = pred_lineage == lin
        is_true = true_lineage == lin
        n_true = int(is_true.sum())

        for t in THRESHOLDS:
            keep = is_pred & (max_lineage_proba >= t)
            n_acc = int(keep.sum())

            if n_acc == 0:
                precision = np.nan
                recall = 0.0
            else:
                tp = int((keep & is_true).sum())
                precision = tp / n_acc
                recall = tp / n_true if n_true > 0 else np.nan

            rows.append({
                "lineage": lin,
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "accepted_rate": n_acc / len(y_true),
                "n_accepted": n_acc,
            })

    curves = pd.DataFrame(rows)
    curves.to_csv(os.path.join(OUTDIR, "lineage_pr_curves.csv"), index=False)

    chosen = []
    for lin, grp in curves.groupby("lineage"):
        cand = grp[grp["precision"] >= PRECISION_TARGET].copy()

        if not cand.empty:
            cand = cand.sort_values(
                ["recall", "accepted_rate", "threshold"],
                ascending=[False, False, True],
            )
            best = cand.iloc[0]
            status = "meets_target"
        else:
            tmp = grp.copy()
            tmp["precision"] = tmp["precision"].fillna(0.0)
            tmp = tmp.sort_values(
                ["precision", "recall", "accepted_rate", "threshold"],
                ascending=[False, False, False, True],
            )
            best = tmp.iloc[0]
            status = "fallback_max_precision"

        chosen.append({
            "lineage": lin,
            "chosen_threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "accepted_rate": float(best["accepted_rate"]),
            "n_accepted": int(best["n_accepted"]),
            "status": status,
        })

    chosen = pd.DataFrame(chosen)
    chosen.to_csv(os.path.join(OUTDIR, "lineage_thresholds.csv"), index=False)

    thr_map = dict(zip(chosen["lineage"], chosen["chosen_threshold"]))

    lineage_accept = np.array([
        max_lineage_proba[i] >= thr_map.get(pred_lineage[i], 1.0)
        for i in range(len(y_true))
    ])

    decision = np.where(lineage_accept, pred_lineage, "Unknown")

    out = pd.DataFrame({
        "true": y_true,
        "true_lineage": true_lineage,
        "pred_lineage": pred_lineage,
        "decision_lineage": decision,
        "lineage_confidence": max_lineage_proba,
    })
    out.to_csv(os.path.join(OUTDIR, "lineage_decision_table.csv"), index=False)

    coverage = float(lineage_accept.mean())
    accuracy = float((true_lineage[lineage_accept] == pred_lineage[lineage_accept]).mean())

    with open(os.path.join(OUTDIR, "lineage_decision_summary.txt"), "w") as f:
        f.write(f"Lineage precision target: {PRECISION_TARGET:.3f}\n")
        f.write(chosen.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n\nOverall accepted lineage decisions:\n")
        f.write(f"  coverage: {coverage:.4f}\n")
        f.write(f"  accuracy: {accuracy:.4f}\n")

    print("Lineage threshold analysis complete.")
    print(f"Coverage: {coverage:.3f}")
    print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
