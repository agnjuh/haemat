import os
import numpy as np
import pandas as pd
import scanpy as sc

OUTDIR = "results/qc"
os.makedirs(OUTDIR, exist_ok=True)

DATA = "data/pbmc10k.h5ad"
DECISION = "results/cross_dataset/decision_table.csv"
PROGRAMS = "results/program_scores/program_scores.csv"
DISPLACEMENT = "results/dynamics/identity_displacement.csv"

PLATELET_CUTOFF_Q = 0.95
HIGH_ENTROPY_Q = 0.95
LOW_GAP_Q = 0.05
HIGH_DISPLACEMENT_CUTOFF = 0.2

adata = sc.read_h5ad(DATA)
decision = pd.read_csv(DECISION)
programs = pd.read_csv(PROGRAMS)
disp = pd.read_csv(DISPLACEMENT)

df = decision.copy()
df["cell_index"] = np.arange(len(df))

if "cell_id" in programs.columns:
    programs = programs.drop(columns=["cell_id"])

df = pd.concat([df, programs.reset_index(drop=True)], axis=1)
df = pd.concat([df, disp.add_prefix("disp_").reset_index(drop=True)], axis=1)

df["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
df["n_genes"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

mt_genes = [g for g in adata.var_names if g.upper().startswith("MT-")]
if mt_genes:
    mt_counts = np.asarray(adata[:, mt_genes].X.sum(axis=1)).ravel()
    total_counts = np.asarray(adata.X.sum(axis=1)).ravel()
    df["pct_mito"] = np.divide(
        mt_counts, total_counts,
        out=np.zeros_like(mt_counts, dtype=float),
        where=total_counts != 0
    )
else:
    df["pct_mito"] = np.nan

df["is_uncertain"] = (
    (df["entropy"] >= df["entropy"].quantile(HIGH_ENTROPY_Q)) |
    (df["top2_gap"] <= df["top2_gap"].quantile(LOW_GAP_Q)) |
    (df["decision"] == "Unknown") |
    (df["disp_displacement"] > HIGH_DISPLACEMENT_CUTOFF)
)

df["platelet_program_high"] = (
    df["Platelet_program"] >= df["Platelet_program"].quantile(PLATELET_CUTOFF_Q)
)

df["high_counts"] = df["n_counts"] >= df["n_counts"].quantile(0.98)
df["high_genes"] = df["n_genes"] >= df["n_genes"].quantile(0.98)
df["low_counts"] = df["n_counts"] <= df["n_counts"].quantile(0.05)
df["low_genes"] = df["n_genes"] <= df["n_genes"].quantile(0.05)

df["cytotoxic_high"] = df["NK_cytotoxic_program"] >= df["NK_cytotoxic_program"].quantile(0.90)
df["myeloid_high"] = df["Myeloid_program"] >= df["Myeloid_program"].quantile(0.90)
df["b_high"] = df["B_program"] >= df["B_program"].quantile(0.90)
df["t_high"] = df["T_program"] >= df["T_program"].quantile(0.90)

df["possible_doublet"] = (
    (df["high_counts"] | df["high_genes"]) &
    (
        (df["cytotoxic_high"] & df["myeloid_high"]) |
        (df["b_high"] & df["myeloid_high"]) |
        (df["t_high"] & df["myeloid_high"]) |
        (df["platelet_program_high"] & (df["cytotoxic_high"] | df["myeloid_high"] | df["t_high"] | df["b_high"]))
    )
)

df["platelet_artifact_suspect"] = (
    df["platelet_program_high"] |
    (df["disp_top1"] == "Platelet") |
    (df["disp_top2"] == "Platelet")
)

df["low_quality_uncertain"] = (
    df["is_uncertain"] &
    (df["low_counts"] | df["low_genes"])
)

df["clean_biological_ambiguity_candidate"] = (
    df["is_uncertain"] &
    ~df["possible_doublet"] &
    ~df["platelet_artifact_suspect"] &
    ~df["low_quality_uncertain"]
)

def status(row):
    if not row["is_uncertain"]:
        return "not_uncertain"
    if row["low_quality_uncertain"]:
        return "low_quality_uncertain"
    if row["possible_doublet"]:
        return "possible_doublet"
    if row["platelet_artifact_suspect"]:
        return "platelet_artifact_suspect"
    return "clean_biological_ambiguity_candidate"

def reason(row):
    reasons = []
    if row["decision"] == "Unknown":
        reasons.append("unknown_decision")
    if row["entropy"] >= df["entropy"].quantile(HIGH_ENTROPY_Q):
        reasons.append("high_entropy")
    if row["top2_gap"] <= df["top2_gap"].quantile(LOW_GAP_Q):
        reasons.append("low_top2_gap")
    if row["disp_displacement"] > HIGH_DISPLACEMENT_CUTOFF:
        reasons.append("high_displacement")
    if row["platelet_artifact_suspect"]:
        reasons.append("platelet_signal_or_platelet_top2")
    if row["possible_doublet"]:
        reasons.append("mixed_program_high_counts_doublet_like")
    if row["low_quality_uncertain"]:
        reasons.append("low_counts_or_low_genes")
    return ";".join(reasons) if reasons else "stable_prediction"

df["artifact_status"] = df.apply(status, axis=1)
df["artifact_reason"] = df.apply(reason, axis=1)
df["use_for_biological_interpretation"] = df["artifact_status"].eq(
    "clean_biological_ambiguity_candidate"
)

df.to_csv(f"{OUTDIR}/artifact_flags.csv", index=False)

summary = (
    df["artifact_status"]
    .value_counts()
    .rename_axis("artifact_status")
    .reset_index(name="n_cells")
)
summary["fraction"] = summary["n_cells"] / len(df)
summary.to_csv(f"{OUTDIR}/artifact_summary.csv", index=False)

uncertain_summary = (
    df[df["is_uncertain"]]
    .groupby(["artifact_status", "disp_top1", "disp_top2"])
    .size()
    .reset_index(name="n_cells")
    .sort_values("n_cells", ascending=False)
)
uncertain_summary.to_csv(f"{OUTDIR}/uncertain_artifact_pair_summary.csv", index=False)

print("\nArtifact summary:")
print(summary.to_string(index=False))

print("\nTop uncertain pairs by artifact status:")
print(uncertain_summary.head(30).to_string(index=False))

print(f"\nSaved: {OUTDIR}/artifact_flags.csv")
print(f"Saved: {OUTDIR}/artifact_summary.csv")
print(f"Saved: {OUTDIR}/uncertain_artifact_pair_summary.csv")
