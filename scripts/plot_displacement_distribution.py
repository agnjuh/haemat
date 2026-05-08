import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("results/dynamics/identity_displacement.csv")

subset = df[(df["top1"]=="NK") & (df["top2"]=="Platelet")]

plt.figure(figsize=(5,4))
sns.histplot(subset["displacement"], bins=30)

plt.title("NK → Platelet displacement distribution")
plt.xlabel("Displacement")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("results/dynamics/NK_to_platelet_distribution.png", dpi=300)
plt.close()
