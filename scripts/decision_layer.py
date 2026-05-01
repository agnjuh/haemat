import numpy as np
import pandas as pd

LINEAGE_MAP = {
    "lymphoid": ["T", "NK", "B"],
    "myeloid": ["Mono", "DC"],
    "platelet": ["Platelet"]
}

def entropy(p):
    p = p + 1e-12
    return -np.sum(p * np.log(p))

def main():
    proba = np.load("results/cross_dataset/proba.npy")
    y_true = np.load("results/cross_dataset/y_test.npy", allow_pickle=True)
    classes = np.load("results/cross_dataset/classes.npy", allow_pickle=True)

    df_thr = pd.read_csv("results/cross_dataset/per_class_thresholds.csv")
    thr_map = dict(zip(df_thr["class"], df_thr["chosen_threshold"]))

    class_to_idx = {c: i for i, c in enumerate(classes)}

    rows = []

    for i in range(len(y_true)):
        p = proba[i]

        # basic stats
        max_idx = np.argmax(p)
        max_class = classes[max_idx]
        max_prob = p[max_idx]

        sorted_p = np.sort(p)[::-1]
        top2_gap = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else 0.0

        ent = entropy(p)

        # lineage scores
        lineage_scores = {}
        for lin, clist in LINEAGE_MAP.items():
            idxs = [class_to_idx[c] for c in clist if c in class_to_idx]
            lineage_scores[lin] = float(p[idxs].sum()) if idxs else 0.0

        best_lineage = max(lineage_scores, key=lineage_scores.get)
        lineage_conf = lineage_scores[best_lineage]

        # restrict to lineage
        candidates = LINEAGE_MAP[best_lineage]

        best_class = None
        best_prob = -1

        for c in candidates:
            if c not in class_to_idx:
                continue
            ci = class_to_idx[c]
            if p[ci] > best_prob:
                best_prob = p[ci]
                best_class = c

        # threshold
        t = thr_map.get(best_class, 1.0)

        if best_prob >= t:
            decision = best_class
        else:
            decision = "Unknown"

        rows.append({
            "true": y_true[i],
            "pred": max_class,
            "decision": decision,
            "max_proba": float(max_prob),
            "top2_gap": float(top2_gap),
            "entropy": float(ent),
            "lineage": best_lineage,
            "lineage_confidence": float(lineage_conf)
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/cross_dataset/decision_table.csv", index=False)

    accepted = df[df["decision"] != "Unknown"]

    acc = (accepted["decision"] == accepted["true"]).mean()
    cov = len(accepted) / len(df)

    with open("results/cross_dataset/decision_summary.txt", "w") as f:
        f.write(f"Coverage: {cov:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")

    print("Decision layer (uncertainty-aware) applied.")
    print(f"Coverage: {cov:.3f}")
    print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
