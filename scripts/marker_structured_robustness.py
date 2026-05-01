import os
import shutil
import subprocess
import pandas as pd

ORIG = "markers_curated.csv"
BACKUP = "markers_curated_backup_structured.csv"
OUTDIR = "results/robustness"
os.makedirs(OUTDIR, exist_ok=True)

def run(cmd):
    print(f"\n[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def evaluate(tag):
    summary = open("results/cross_dataset/decision_summary.txt").read()
    cov = float(summary.split("Coverage:")[1].split("\n")[0])
    acc = float(summary.split("Accuracy:")[1].split("\n")[0])

    report = open("results/cross_dataset/report_cross_logreg.txt").read()
    macro_f1 = float(report.split("macro-F1=")[1].split("\n")[0])

    return {
        "test": tag,
        "coverage": cov,
        "accuracy": acc,
        "macro_f1": macro_f1
    }

def main():
    shutil.copy(ORIG, BACKUP)
    df = pd.read_csv(ORIG)

    results = []

    try:
        # --- 1. core only ---
        df_core = df[df["tier"] == "core"]
        df_core.to_csv(ORIG, index=False)

        run("python scripts/cross_dataset.py")
        run("python scripts/analyze_thresholds_per_class.py --outdir results/cross_dataset --precision-target 0.95")
        run("python scripts/decision_layer.py")

        results.append(evaluate("core_only"))

        # --- 2. remove lineage markers ---
        lineage_genes = {
            "CD3D","CD3E","CD3G","TRAC",   # T
            "NKG7","GNLY",                # NK
            "MS4A1","CD79A",              # B
            "LYZ","S100A8","S100A9"       # Mono
        }

        df_no_lineage = df[~df["gene"].isin(lineage_genes)]
        df_no_lineage.to_csv(ORIG, index=False)

        run("python scripts/cross_dataset.py")
        run("python scripts/analyze_thresholds_per_class.py --outdir results/cross_dataset --precision-target 0.95")
        run("python scripts/decision_layer.py")

        results.append(evaluate("no_lineage"))

        # --- 3. remove state markers ---
        state_genes = {
            "IL7R","CCR7","TCF7"
        }

        df_no_state = df[~df["gene"].isin(state_genes)]
        df_no_state.to_csv(ORIG, index=False)

        run("python scripts/cross_dataset.py")
        run("python scripts/analyze_thresholds_per_class.py --outdir results/cross_dataset --precision-target 0.95")
        run("python scripts/decision_layer.py")

        results.append(evaluate("no_state"))

    finally:
        shutil.copy(BACKUP, ORIG)

    out = pd.DataFrame(results)
    out.to_csv(f"{OUTDIR}/marker_structured_robustness.csv", index=False)
    print("\nSaved → results/robustness/marker_structured_robustness.csv")

if __name__ == "__main__":
    main()
