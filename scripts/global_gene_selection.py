import pandas as pd
import numpy as np
import os

IN = "results/top5_union_heatmap.csv"
OUT = "results/global_programs"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(IN, index_col=0)

# variance across cell types
var = df.var(axis=1)

top = var.sort_values(ascending=False).head(20)

top.to_csv(f"{OUT}/top_variable_genes.csv")

print("Saved:", f"{OUT}/top_variable_genes.csv")
