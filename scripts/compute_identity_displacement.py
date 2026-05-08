import numpy as np
import pandas as pd
import os

PROBA = "results/cross_dataset/proba.npy"
CLASSES = "results/cross_dataset/classes.npy"
OUT = "results/dynamics/identity_displacement.csv"

os.makedirs("results/dynamics", exist_ok=True)

proba = np.load(PROBA)
classes = np.load(CLASSES, allow_pickle=True)

rows = []

for i in range(proba.shape[0]):
    p = proba[i]

    idx = np.argsort(p)[::-1]

    top1 = classes[idx[0]]
    top2 = classes[idx[1]]

    p1 = p[idx[0]]
    p2 = p[idx[1]]

    gap = p1 - p2
    displacement_strength = 1 - gap  # small gap = strong displacement

    rows.append({
        "top1": top1,
        "top2": top2,
        "p1": p1,
        "p2": p2,
        "gap": gap,
        "displacement": displacement_strength
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print("Saved:", OUT)
