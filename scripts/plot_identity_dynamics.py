import pandas as pd
import matplotlib.pyplot as plt
import os

IN_TABLE = "results/dynamics/identity_dynamics_table.csv"
IN_MATRIX = "results/dynamics/transition_matrix.csv"

OUTDIR = "results/dynamics"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(IN_TABLE)
tm = pd.read_csv(IN_MATRIX, index_col=0)

# ----------------------------
# Transition heatmap
# ----------------------------
plt.figure()
plt.imshow(tm.values)
plt.xticks(range(len(tm.columns)), tm.columns, rotation=45)
plt.yticks(range(len(tm.index)), tm.index)
plt.colorbar()
plt.title("Transition matrix (top1 → second)")
plt.tight_layout()

plt.savefig(f"{OUTDIR}/transition_matrix.png", dpi=300)
plt.close()

# ----------------------------
# Stability landscape
# ----------------------------
plt.figure()
plt.scatter(df["entropy_normalized"], df["stability_score"], alpha=0.3)
plt.xlabel("Entropy (normalized)")
plt.ylabel("Stability score")
plt.title("Identity stability landscape")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/stability_landscape.png", dpi=300)
plt.close()

print("Plots saved.")
