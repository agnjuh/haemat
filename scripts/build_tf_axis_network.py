import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

INFILE = "results/regulatory/tf_enrichment/tf_enrichment_all_axes.csv"
OUTDIR = "results/regulatory/tf_network"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INFILE)

# Keep exploratory but biologically relevant hits
# Small gene sets rarely survive FDR, --> keeping nominal signal and annotate it
df = df[df["P-value"] <= 0.05].copy()

def extract_tf(term):
    # First token is usually TF name, e.g. "SPI1 23547873 ChIP-Seq NB4 Human"
    return str(term).split()[0]

df["TF"] = df["Term"].apply(extract_tf)

# Remove generic / less interpretable factors for this specific biological story
exclude = {
    "CTCF", "OCT4", "EZH2", "MAX", "NFYB", "REST", "CBX3",
    "eGFP-FOS", "ERA"
}
df = df[~df["TF"].isin(exclude)].copy()

# Keep relevant immune / hematopoietic / inflammatory TFs if present
priority_tfs = {
    "SPI1", "PU.1", "IRF1", "IRF8", "STAT4", "STAT5A",
    "RELA", "RUNX1", "GATA2", "GATA3", "FLI1",
    "LMO2", "LYL1", "MECOM", "CEBPB", "NR1H3", "PPARG",
    "BACH1", "MEIS1", "SCL", "ETS1"
}

df = df[df["TF"].isin(priority_tfs)].copy()

# Summarise TF-axis links
edges = (
    df.groupby(["axis", "TF"])
      .agg(
          min_p=("P-value", "min"),
          min_adj_p=("Adjusted P-value", "min"),
          max_combined_score=("Combined Score", "max"),
          libraries=("library", lambda x: ";".join(sorted(set(x)))),
          terms=("Term", lambda x: " | ".join(sorted(set(map(str, x)))[:3])),
          genes=("Genes", lambda x: ";".join(sorted(set(";".join(map(str, x)).split(";")))))
      )
      .reset_index()
)

edges = edges.sort_values(["axis", "min_p", "max_combined_score"], ascending=[True, True, False])
edges.to_csv(f"{OUTDIR}/tf_axis_edges.csv", index=False)

print("Saved:", f"{OUTDIR}/tf_axis_edges.csv")
print(edges)

# Build network
G = nx.Graph()

axis_nodes = sorted(edges["axis"].unique())
tf_nodes = sorted(edges["TF"].unique())

for axis in axis_nodes:
    G.add_node(axis, kind="axis")

for tf in tf_nodes:
    G.add_node(tf, kind="tf")

for _, r in edges.iterrows():
    weight = max(1.0, min(8.0, r["max_combined_score"] / 20.0))
    G.add_edge(r["axis"], r["TF"], weight=weight, label=f'p={r["min_p"]:.3g}')

# Layout: axis nodes fixed on left, TFs on right
pos = {}

axis_y = list(range(len(axis_nodes)))
for i, axis in enumerate(axis_nodes):
    pos[axis] = (0, -i)

for i, tf in enumerate(tf_nodes):
    pos[tf] = (2.5, -i * (max(1, len(axis_nodes)) / max(1, len(tf_nodes))))

plt.figure(figsize=(10, 7))

axis_list = [n for n, d in G.nodes(data=True) if d["kind"] == "axis"]
tf_list = [n for n, d in G.nodes(data=True) if d["kind"] == "tf"]

nx.draw_networkx_nodes(
    G, pos,
    nodelist=axis_list,
    node_size=2600,
    node_shape="s",
    edgecolors="black",
    linewidths=1.2
)

nx.draw_networkx_nodes(
    G, pos,
    nodelist=tf_list,
    node_size=1500,
    node_shape="o",
    edgecolors="black",
    linewidths=1.0
)

widths = [G[u][v]["weight"] for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=widths, alpha=0.65)

nx.draw_networkx_labels(G, pos, font_size=8)

edge_labels = {(u, v): G[u][v]["label"] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title("Exploratory TF-axis regulatory network")
plt.axis("off")
plt.tight_layout()

plt.savefig(f"{OUTDIR}/tf_axis_network.png", dpi=300)
plt.savefig(f"{OUTDIR}/tf_axis_network.pdf")
plt.close()

print("Saved:")
print(f"{OUTDIR}/tf_axis_network.png")
print(f"{OUTDIR}/tf_axis_network.pdf")
