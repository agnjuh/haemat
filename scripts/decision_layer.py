import numpy as np
import pandas as pd
import os

OUTDIR = "results/cross_dataset"

proba = np.load(os.path.join(OUTDIR, "proba.npy"))
y_true = np.load(os.path.join(OUTDIR, "y_test.npy"), allow_pickle=True)
classes = np.load(os.path.join(OUTDIR, "classes.npy"), allow_pickle=True)

thresholds = pd.read_csv(os.path.join(OUTDIR, "per_class_thresholds.csv"))

# map thresholds
thr_map = dict(zip(thresholds["class"], thresholds["chosen_threshold"]))

# predictions
argmax_idx = proba.argmax(axis=1)
y_pred = classes[argmax_idx]
maxp = proba.max(axis=1)

# decision
y_decision = []
accepted_mask = []

for i in range(len(y_pred)):
    c = y_pred[i]
    p = proba[i, argmax_idx[i]]
    t = thr_map.get(c, 1.0)

    if p >= t:
        y_decision.append(c)
        accepted_mask.append(True)
    else:
        y_decision.append("Unknown")
        accepted_mask.append(False)

y_decision = np.array(y_decision)
accepted_mask = np.array(accepted_mask)

# save
pd.DataFrame({
    "true": y_true,
    "pred": y_pred,
    "decision": y_decision,
    "accepted": accepted_mask,
    "confidence": maxp
}).to_csv(os.path.join(OUTDIR, "decision_table.csv"), index=False)

# metrics
coverage = accepted_mask.mean()

if accepted_mask.sum() > 0:
    acc = (y_true[accepted_mask] == y_pred[accepted_mask]).mean()
else:
    acc = np.nan

with open(os.path.join(OUTDIR, "decision_summary.txt"), "w") as f:
    f.write(f"Coverage: {coverage:.4f}\n")
    f.write(f"Accuracy (accepted only): {acc:.4f}\n")

print("Decision layer applied.")
print(f"Coverage: {coverage:.3f}")
print(f"Accuracy: {acc:.3f}")
