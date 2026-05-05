import os
import numpy as np
import pandas as pd

OUTDIR = "results/transitional_states"
os.makedirs(OUTDIR, exist_ok=True)

DECISION = "results/cross_dataset/decision_table.csv"
SCORES = "results/program_scores/program_scores.csv"

decision = pd.read_csv(DECISION)
scores = pd.read_csv(SCORES)

df = decision.copy()
score_cols = [c for c in scores.columns if c != "cell_id"]

for col in score_cols:
    df[col] = scores[col].values

required = {"entropy", "top2_gap", "decision"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in decision table: {missing}")


def dominant_program(row):
    vals = {
        "B": row.get("B_program", np.nan),
        "T": row.get("T_program", np.nan),
        "NK": row.get("NK_cytotoxic_program", np.nan),
        "Mono": row.get("Myeloid_program", np.nan),
        "DC": row.get("DC_program", np.nan),
        "Platelet": row.get("Platelet_program", np.nan),
    }
    vals = {k: v for k, v in vals.items() if pd.notna(v)}
    if not vals:
        return "Unknown"
    return max(vals, key=vals.get)


def ratio(a, b):
    if max(a, b) <= 0:
        return 0
    return min(a, b) / max(a, b)


def classify_transitional_state(row):
    decision_label = row["decision"]
    entropy = row["entropy"]
    top2_gap = row["top2_gap"]

    t_score = row.get("T_program", 0)
    nk_score = row.get("NK_cytotoxic_program", 0)
    myeloid_score = row.get("Myeloid_program", 0)
    dc_score = row.get("DC_program", 0)
    platelet_score = row.get("Platelet_program", 0)

    t_nk_ratio = ratio(t_score, nk_score)
    myeloid_dc_ratio = ratio(myeloid_score, dc_score)

    if decision_label == "Unknown":
        return "unknown_low_confidence"

    if entropy >= 0.5 and top2_gap <= 0.5:
        return "high_uncertainty"

    # T/NK ambiguity should require mixed cytotoxic/T signal AND model ambiguity.
    # This avoids calling all cytotoxic T cells transitional.
    if decision_label in ["T", "NK"]:
        if t_score > 0 and nk_score > 0:
            if t_nk_ratio >= 0.35 and (top2_gap <= 0.85 or entropy >= 0.10):
                return "cytotoxic_transitional"

    # Cytotoxic but still confident T-like state.
    if decision_label == "T":
        if nk_score > 0 and t_score > 0 and t_nk_ratio >= 0.35:
            return "confident_cytotoxic_T_like"

    # Myeloid/DC ambiguity.
    if decision_label in ["Mono", "DC"]:
        if myeloid_score > 0 and dc_score > 0:
            if myeloid_dc_ratio >= 0.25:
                return "myeloid_DC_transitional"

    # Platelet-like ambiguity in non-platelet decisions.
    if decision_label != "Platelet":
        other_max = max(t_score, nk_score, myeloid_score, dc_score, 1e-9)
        if platelet_score > 0 and platelet_score >= other_max:
            return "platelet_like_ambiguous"

    return "confident"


df["dominant_program"] = df.apply(dominant_program, axis=1)
df["T_NK_program_ratio"] = df.apply(
    lambda r: ratio(r.get("T_program", 0), r.get("NK_cytotoxic_program", 0)),
    axis=1
)
df["Myeloid_DC_program_ratio"] = df.apply(
    lambda r: ratio(r.get("Myeloid_program", 0), r.get("DC_program", 0)),
    axis=1
)
df["transitional_state"] = df.apply(classify_transitional_state, axis=1)

summary = (
    df["transitional_state"]
    .value_counts()
    .rename_axis("transitional_state")
    .reset_index(name="n_cells")
)

by_decision = pd.crosstab(df["decision"], df["transitional_state"]).reset_index()
by_true = pd.crosstab(df["true"], df["transitional_state"]).reset_index()

df.to_csv(f"{OUTDIR}/transitional_state_table.csv", index=False)
summary.to_csv(f"{OUTDIR}/transitional_summary.csv", index=False)
by_decision.to_csv(f"{OUTDIR}/transitional_by_decision.csv", index=False)
by_true.to_csv(f"{OUTDIR}/transitional_by_true.csv", index=False)

print("Saved:")
print(f"{OUTDIR}/transitional_state_table.csv")
print(f"{OUTDIR}/transitional_summary.csv")
print(f"{OUTDIR}/transitional_by_decision.csv")
print(f"{OUTDIR}/transitional_by_true.csv")
print()
print(summary)
