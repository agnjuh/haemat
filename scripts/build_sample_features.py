import os
import pandas as pd
import numpy as np

OUTDIR = "results/sample_level"
os.makedirs(OUTDIR, exist_ok=True)

TABLE = "results/transitional_states/transitional_state_table.csv"

df = pd.read_csv(TABLE)

# sample id (egyelőre egy minta)
df["sample_id"] = "PBMC10k"

features = []

for sample, sub in df.groupby("sample_id"):
    total = len(sub)

    row = {"sample_id": sample}

    # 1. transitional state fractions
    state_counts = sub["transitional_state"].value_counts()

    for state, count in state_counts.items():
        row[f"frac_{state}"] = count / total

    # ensure missing states are present
    all_states = [
        "confident",
        "confident_cytotoxic_T_like",
        "cytotoxic_transitional",
        "myeloid_DC_transitional",
        "platelet_like_ambiguous",
        "high_uncertainty",
        "unknown_low_confidence",
    ]

    for s in all_states:
        if f"frac_{s}" not in row:
            row[f"frac_{s}"] = 0.0

    # 2. uncertainty
    row["mean_entropy"] = sub["entropy"].mean()
    row["median_entropy"] = sub["entropy"].median()
    row["mean_top2_gap"] = sub["top2_gap"].mean()

    # 3. program scores
    program_cols = [
        "B_program",
        "T_program",
        "NK_cytotoxic_program",
        "Myeloid_program",
        "DC_program",
        "Platelet_program",
    ]

    for col in program_cols:
        if col in sub.columns:
            row[f"mean_{col}"] = sub[col].mean()

    # 4. ambiguity load
    row["fraction_non_confident"] = 1 - row["frac_confident"]

    features.append(row)

features_df = pd.DataFrame(features)

features_df.to_csv(f"{OUTDIR}/sample_features.csv", index=False)

print("Saved:")
print(f"{OUTDIR}/sample_features.csv")
print()
print(features_df)
