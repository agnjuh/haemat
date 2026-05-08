import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

INFILE = "results/regulatory/tf_network/tf_axis_edges.csv"
OUTDIR = "results/regulatory/tf_network"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INFILE)

keep = {
    "Cytotoxic T/NK": ("cytotoxic_T_NK", ["RELA", "FLI1", "GATA3"]),
    "Myeloid/DC": ("myeloid_DC", ["SPI1", "NR1H3", "RUNX1"]),
    "Platelet-associated": ("platelet_associated", ["SPI1", "RUNX1", "GATA2", "LMO2", "LYL1", "CEBPB"]),
}

rows = []
for display_axis, (axis, tfs) in keep.items():
    sub = df[(df["axis"] == axis) & (df["TF"].isin(tfs))].copy()
    sub["display_axis"] = display_axis
    rows.append(sub)

plot_df = pd.concat(rows, ignore_index=True)
plot_df.to_csv(f"{OUTDIR}/key_tf_axis_edges.csv", index=False)

G = nx.Graph()

axis_nodes = list(keep.keys())
tf_nodes = sorted(plot_df["TF"].unique())

for axis in axis_nodes:
    G.add_node(axis, kind="axis")

for tf in tf_nodes:
    G.add_node(tf, kind="tf")

for _, r in plot_df.iterrows():
    weight = max(1.2, min(6.0, r["max_combined_score"] / 25.0))
    G.add_edge(r["display_axis"], r["TF"], weight=weight)

pos = {
    "Cytotoxic T/NK": (0, 1.4),
    "Myeloid/DC": (0, 0),
    "Platelet-associated": (0, -1.4),
}

tf_y = {
    "CEBPB": 2.0,
    "FLI1": 1.5,
    "GATA2": 1.0,
    "GATA3": 0.5,
    "LMO2": 0.0,
    "LYL1": -0.5,
    "NR1H3": -1.0,
    "RELA": -1.5,
    "RUNX1": -2.0,
    "SPI1": -2.5,
}

for tf in tf_nodes:
    pos[tf] = (3.2, tf_y.get(tf, 0))

plt.figure(figsize=(10, 6))

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=axis_nodes,
    node_shape="s",
    node_size=3600,
    edgecolors="black",
    linewidths=1.2
)

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=tf_nodes,
    node_shape="o",
    node_size=1900,
    edgecolors="black",
    linewidths=1.0
)

widths = [G[u][v]["weight"] for u, v in G.edges()]

nx.draw_networkx_edges(
    G,
    pos,
    width=widths,
    alpha=0.65
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=9,
    font_weight="bold"
)

plt.title("Key TF-axis regulatory network")
plt.axis("off")
plt.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.08)

plt.savefig(f"{OUTDIR}/key_tf_axis_network_clean.png", dpi=300)
plt.savefig(f"{OUTDIR}/key_tf_axis_network_clean.pdf")
plt.close()

print("Saved:")
print(f"{OUTDIR}/key_tf_axis_network_clean.png")
print(f"{OUTDIR}/key_tf_axis_network_clean.pdf")
