import numpy as np
import pandas as pd

# lineage definition
LINEAGE_MAP = {
    "lymphoid": ["T", "NK", "B"],
    "myeloid": ["Mono", "DC"],
    "platelet": ["Platelet"]
}

def main():
    proba = np.load("results/cross_dataset/proba.npy")
    y_true = np.load("results/cross_dataset/y_test.npy", allow_pickle=True)
    classes = np.load("results/cross_dataset/classes.npy", allow_pickle=True)

    df_thr = pd.read_csv("results/cross_dataset/per_class_thresholds.csv")
    thr_map = dict(zip(df_thr["class"], df_thr["chosen_threshold"]))

    class_to_idx = {c: i for i, c in enumerate(classes)}

    decisions = []

    for i in range(len(y_true)):
        p = proba[i]

        # --- lineage scores ---
        lineage_scores = {}
        for lin, clist in LINEAGE_MAP.items():
            idxs = [class_to_idx[c] for c in clist if c in class_to_idx]
            lineage_scores[lin] = float(p[idxs].sum()) if idxs else 0.0

        best_lineage = max(lineage_scores, key=lineage_scores.get)

        # --- restrict to lineage ---
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

        # --- threshold ---
        t = thr_map.get(best_class, 1.0)

        if best_prob >= t:
            decision = best_class
        else:
            decision = "Unknown"

        decisions.append({
            "true": y_true[i],
            "pred": classes[np.argmax(p)],
            "decision": decision,
            "max_proba": float(best_prob),
            "lineage": best_lineage
        })

    df = pd.DataFrame(decisions)
    df.to_csv("results/cross_dataset/decision_table.csv", index=False)

    acc = (df[df["decision"] != "Unknown"]["decision"] == df[df["decision"] != "Unknown"]["true"]).mean()
    cov = (df["decision"] != "Unknown").mean()

    with open("results/cross_dataset/decision_summary.txt", "w") as f:
        f.write(f"Coverage: {cov:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")

    print("Decision layer (lineage-aware) applied.")
    print(f"Coverage: {cov:.3f}")
    print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
