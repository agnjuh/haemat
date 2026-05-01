#!/usr/bin/env python3
"""
Per-class operating points for scRNA-seq classifier.

Inputs expected in --outdir (produced by your pipeline):
  - proba.npy      : (n_samples, n_classes) class probabilities
  - y_test.npy     : (n_samples,) true labels (dtype object/str)
  - classes.npy    : (n_classes,) class labels corresponding to proba columns

Outputs (written to --outdir):
  - per_class_thresholds.csv
  - per_class_pr_curves.csv                     (long format: class,threshold,precision,recall,accepted_rate,n_accepted)
  - per_class_curves_[CLASS].png                (one plot per class: precision/recall vs threshold)
  - accepted_confusion_matrix_per_class.csv     (using class-specific thresholds)
  - summary_per_class.txt
"""

import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_arrays(outdir):
    proba   = np.load(os.path.join(outdir, "proba.npy"))
    y_true  = np.load(os.path.join(outdir, "y_test.npy"), allow_pickle=True)
    classes = np.load(os.path.join(outdir, "classes.npy"), allow_pickle=True)
    return proba, y_true, classes

def eval_per_class_curves(proba, y_true, classes, thresholds):
    """Compute precision/recall/accepted rate for each class across thresholds."""
    argmax_idx = proba.argmax(axis=1)
    argmax_lab = classes[argmax_idx]
    curves = []
    for ci, c in enumerate(classes):
        p_c = proba[:, ci]
        is_pred_c = (argmax_idx == ci)            # predicted class (winner) is c
        is_true_c = (y_true == c)
        n_true_c  = int(is_true_c.sum())

        for t in thresholds:
            keep = is_pred_c & (p_c >= t)         # accepted as class c at threshold t
            n_acc = int(keep.sum())
            if n_acc == 0:
                prec = np.nan; rec = 0.0
            else:
                tp = int((keep & is_true_c).sum())
                prec = tp / n_acc
                rec  = (tp / n_true_c) if n_true_c > 0 else np.nan
            acc_rate = n_acc / len(y_true)        # fraction of the *dataset* accepted as class c
            curves.append((c, t, prec, rec, acc_rate, n_acc))
    df = pd.DataFrame(curves, columns=["class","threshold","precision","recall","accepted_rate","n_accepted"])
    return df, argmax_lab

def pick_thresholds(df, precision_target):
    """For each class: choose highest-recall threshold among those with precision ≥ target.
       If none meet target, pick the threshold with highest precision (ties → higher recall, then lower threshold)."""
    rows = []
    for c, grp in df.groupby("class"):
        cand = grp[grp["precision"] >= precision_target].copy()
        if not cand.empty:
            cand.sort_values(["recall","accepted_rate","threshold"],
                             ascending=[False,False,True], inplace=True)
            best = cand.iloc[0]
            status = "meets_target"
        else:
            # fallback: best precision available
            grp2 = grp.copy()
            grp2["precision"] = grp2["precision"].fillna(0.0)
            grp2.sort_values(["precision","recall","accepted_rate","threshold"],
                             ascending=[False,False,False,True], inplace=True)
            best = grp2.iloc[0]
            status = "fallback_max_precision"
        rows.append({
            "class": c,
            "chosen_threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "accepted_rate": float(best["accepted_rate"]),
            "n_accepted": int(best["n_accepted"]),
            "status": status
        })
    return pd.DataFrame(rows)

def accepted_mask_with_per_class_thresholds(proba, classes, argmax_lab, per_class_tbl):
    """Build overall accepted mask given class-specific thresholds."""
    thr_map = {r["class"]: r["chosen_threshold"] for _, r in per_class_tbl.iterrows()}
    acc = np.zeros(len(argmax_lab), dtype=bool)
    for ci, c in enumerate(classes):
        t = thr_map.get(c, 1.0)
        acc |= ((argmax_lab == c) & (proba[:, ci] >= t))
    return acc

def plot_curves_per_class(df, outdir):
    for c, grp in df.groupby("class"):
        plt.figure(figsize=(7,4.2))
        plt.plot(grp["threshold"], grp["precision"], label="precision")
        plt.plot(grp["threshold"], grp["recall"], label="recall")
        plt.xlabel("threshold on P(class)")
        plt.ylabel("metric")
        plt.title(f"Per-class curves — {c}")
        plt.legend()
        plt.tight_layout()
        fp = os.path.join(outdir, f"per_class_curves_{c}.png")
        plt.savefig(fp, dpi=300)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results", help="Directory with .npy inputs and where outputs are saved")
    ap.add_argument("--precision-target", type=float, default=0.99, help="Target precision per class")
    ap.add_argument("--tmin", type=float, default=0.50)
    ap.add_argument("--tmax", type=float, default=0.95)
    ap.add_argument("--tstep", type=float, default=0.01)
    args = ap.parse_args()

    # Check inputs
    for f in ["proba.npy","y_test.npy","classes.npy"]:
        if not os.path.exists(os.path.join(args.outdir, f)):
            raise SystemExit(
                f"Missing {f} in {args.outdir}. Ensure your pipeline saved:\n"
                "  np.save('results/proba.npy', proba)\n"
                "  np.save('results/y_test.npy', y_test)\n"
                "  np.save('results/classes.npy', lr.classes_)\n"
            )

    proba, y_true, classes = load_arrays(args.outdir)
    thresholds = np.round(np.arange(args.tmin, args.tmax + 1e-9, args.tstep), 4)

    # Curves
    df_curves, argmax_lab = eval_per_class_curves(proba, y_true, classes, thresholds)
    curves_csv = os.path.join(args.outdir, "per_class_pr_curves.csv")
    df_curves.to_csv(curves_csv, index=False)

    # Pick thresholds
    tbl = pick_thresholds(df_curves, args.precision_target)
    thr_csv = os.path.join(args.outdir, "per_class_thresholds.csv")
    tbl.to_csv(thr_csv, index=False)

    # Plot curves per class
    plot_curves_per_class(df_curves, args.outdir)

    # Accepted-set with per-class thresholds
    acc_mask = accepted_mask_with_per_class_thresholds(proba, classes, argmax_lab, tbl)
    if acc_mask.sum() > 0:
        cm = confusion_matrix(y_true[acc_mask], argmax_lab[acc_mask], labels=classes)
        df_cm = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
        cm_csv = os.path.join(args.outdir, "accepted_confusion_matrix_per_class.csv")
        df_cm.to_csv(cm_csv)
        overall_acc = float((y_true[acc_mask] == argmax_lab[acc_mask]).mean())
        coverage = float(acc_mask.mean())
    else:
        df_cm = pd.DataFrame(0, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
        cm_csv = os.path.join(args.outdir, "accepted_confusion_matrix_per_class.csv")
        df_cm.to_csv(cm_csv)
        overall_acc = float("nan"); coverage = 0.0

    # Summary
    summary = os.path.join(args.outdir, "summary_per_class.txt")
    with open(summary, "w") as f:
        f.write(f"Per-class precision target: {args.precision_target:.3f}\n")
        f.write(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n\nOverall (accepted-set using per-class thresholds):\n")
        f.write(f"  coverage: {coverage:.4f}\n")
        f.write(f"  accuracy: {overall_acc:.4f}\n")
        f.write(f"Confusion matrix → {cm_csv}\n")

    print(f"Wrote: {thr_csv}")
    print(f"Wrote: {curves_csv}")
    print(f"Wrote: {os.path.join(args.outdir, 'per_class_curves_[CLASS].png')}")
    print(f"Accepted-set confusion matrix → {cm_csv}")
    print(f"Summary → {summary}")

if __name__ == "__main__":
    main()