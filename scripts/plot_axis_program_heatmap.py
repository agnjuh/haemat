import pandas as pd
import matplotlib.pyplot as plt
import os

INFILE = "results/dynamics/axis_program_means.csv"
OUT = "results/dynamics/axis_program_heatmap.png"

os.makedirs("results/dynamics", exist_ok=True)

df = pd.read_csv(INFILE)
df = df.set_index("axis")

# order axes manually (clean presentation)
axis_order = [
    "cytotoxic_T_NK_axis",
    "myeloid_DC_axis",
    "platelet_associated_axis",
    "other_axis"
]

df = df.loc[axis_order]

# program order (biological logic)
program_order = [
    "B_program",
    "T_program",
    "NK_cytotoxic_program",
    "Myeloid_program",
    "DC_program",
    "Platelet_program"
]

df = df[program_order]

# plot
plt.figure(figsize=(7, 4))
im = plt.imshow(df.values, aspect="auto")

plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="right")
plt.yticks(range(len(df.index)), df.index)

# colorbar
cbar = plt.colorbar(im)
cbar.set_label("Mean program score")

# annotate values
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        val = df.iloc[i, j]
        plt.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

plt.title("Program composition of identity-displacement axes")
plt.tight_layout()
plt.savefig(OUT, dpi=300)
plt.close()

print("Saved:", OUT)
