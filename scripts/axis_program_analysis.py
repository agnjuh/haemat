import os
import pandas as pd
import matplotlib.pyplot as plt


# input
DISP = "results/dynamics/identity_displacement.csv"
PROG = "results/program_scores/program_scores.csv"

OUTDIR = "results/dynamics"
os.makedirs(OUTDIR, exist_ok=True)

THRESHOLD = 0.2


# load
df = pd.read_csv(DISP)
prog = pd.read_csv(PROG)

# merge (assumes same order)
df = df.copy()
prog = prog.copy()

assert len(df) == len(prog), "Mismatch rows"

merged = pd.concat([df, prog], axis=1)


# axis assignment
def assign_axis(row):
    pair = {row["top1"], row["top2"]}

    if pair == {"T", "NK"}:
        return "cytotoxic_T_NK_axis"

    if pair == {"Mono", "DC"}:
        return "myeloid_DC_axis"

    if "Platelet" in pair:
        return "platelet_associated_axis"

    return "other_axis"

merged["axis"] = merged.apply(assign_axis, axis=1)


# high displacement filter
high = merged[merged["displacement"] > THRESHOLD].copy()


# program columns
program_cols = [
    "B_program",
    "T_program",
    "NK_cytotoxic_program",
    "Myeloid_program",
    "DC_program",
    "Platelet_program"
]


# summary
summary = (
    high.groupby("axis")[program_cols]
        .mean()
        .reset_index()
)

summary.to_csv(f"{OUTDIR}/axis_program_means.csv", index=False)

print("Program means by axis:")
print(summary)


# plot
for axis in summary["axis"]:
    sub = high[high["axis"] == axis]

    means = sub[program_cols].mean()

    plt.figure(figsize=(6,4))
    means.sort_values().plot(kind="barh")
    plt.title(f"{axis} - mean program scores")
    plt.xlabel("Score")
    plt.tight_layout()

    fname = f"{OUTDIR}/{axis}_program_profile.png"
    plt.savefig(fname, dpi=300)
    plt.close()

print("Saved plots per axis.")
