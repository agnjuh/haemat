import os
import shutil
import subprocess
import pandas as pd
import numpy as np

ORIG = "markers_curated.csv"
BACKUP = "markers_curated_backup_for_robustness.csv"
OUTDIR = "results/robustness"
os.makedirs(OUTDIR, exist_ok=True)

LEVELS = [1.0, 0.8, 0.6, 0.4]

def run_cmd(cmd):
    print(f"\n[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # backup
    shutil.copy(ORIG, BACKUP)

    df = pd.read_csv(ORIG)
    results = []

    try:
        for frac in LEVELS:
            print(f"\n=== Fraction: {frac} ===")

            df_sub = (
                df.groupby("cell_type", group_keys=False)
                  .apply(lambda x: x.sample(
                      max(1, int(len(x) * frac)),
                      random_state=42
                  ))
            )

            tmp_file = "markers_curated_tmp.csv"
            df_sub.to_csv(tmp_file, index=False)

            # overwrite active markers
            shutil.copy(tmp_file, ORIG)

            # run pipeline
            run_cmd("python scripts/cross_dataset.py")
            run_cmd("python scripts/analyze_thresholds_per_class.py --outdir results/cross_dataset --precision-target 0.95")
            run_cmd("python scripts/decision_layer.py")

            # read outputs
            summary = open("results/cross_dataset/decision_summary.txt").read()

            # extract numbers
            cov = float(summary.split("Coverage:")[1].split("\n")[0])
            acc = float(summary.split("Accuracy:")[1].split("\n")[0])

            report = open("results/cross_dataset/report_cross_logreg.txt").read()
            macro_f1 = float(report.split("macro-F1=")[1].split("\n")[0])

            results.append({
                "fraction": frac,
                "coverage": cov,
                "accuracy": acc,
                "macro_f1": macro_f1
            })

    finally:
        # restore original markers
        shutil.copy(BACKUP, ORIG)

    pd.DataFrame(results).to_csv(f"{OUTDIR}/marker_robustness.csv", index=False)
    print("\nSaved → results/robustness/marker_robustness.csv")

if __name__ == "__main__":
    main()
