import numpy as np
import pandas as pd
from scipy.stats import entropy
import os

# paths
DECISION_PATH = "results/cross_dataset/decision_table.csv"
PROBA_PATH = "results/cross_dataset/proba.npy"
CLASSES_PATH = "results/cross_dataset/classes.npy"

OUTDIR = "results/dynamics"
os.makedirs(OUTDIR, exist_ok=True)


# load data
print("Loading data...")
df = pd.read_csv(DECISION_PATH)
proba = np.load(PROBA_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

assert proba.shape[0] == df.shape[0], "Mismatch between proba and decision_table"


# entropy
print("Computing entropy...")
ent = entropy(proba.T)
max_ent = np.log(proba.shape[1])
ent_norm = ent / max_ent


# top2 + direction
print("Computing top-2 gap and direction...")
top2_idx = np.argsort(proba, axis=1)[:, -2:]

top1 = top2_idx[:, 1]
top2 = top2_idx[:, 0]

top1_prob = proba[np.arange(len(proba)), top1]
top2_prob = proba[np.arange(len(proba)), top2]

top2_gap = top1_prob - top2_prob

top1_class = classes[top1]
top2_class = classes[top2]


# stability
print("Computing stability score...")
stability = top1_prob * (1 - ent_norm)

# dataframe
df["entropy"] = ent
df["entropy_normalized"] = ent_norm
df["top1_prob"] = top1_prob
df["top2_prob"] = top2_prob
df["top2_gap"] = top2_gap
df["top1_class"] = top1_class
df["second_class"] = top2_class
df["stability_score"] = stability

df["direction"] = np.where(
    df["top2_gap"] < 0.2,
    df["second_class"],
    df["top1_class"]
)


# save table
out_table = f"{OUTDIR}/identity_dynamics_table.csv"
df.to_csv(out_table, index=False)
print("Saved:", out_table)


# transition matrix
print("Building transition matrix...")
transition_counts = pd.crosstab(
    df["top1_class"],
    df["second_class"]
)

transition_matrix = transition_counts.div(
    transition_counts.sum(axis=1),
    axis=0
)

out_matrix = f"{OUTDIR}/transition_matrix.csv"
transition_matrix.to_csv(out_matrix)
print("Saved:", out_matrix)
