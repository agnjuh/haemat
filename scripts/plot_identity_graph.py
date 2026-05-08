import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

INFILE = "results/dynamics/ambiguous_neighbour_matrix_filtered_clean.csv"
OUTDIR = "results/dynamics"
os.makedirs(OUTDIR, exist_ok=True)

m = pd.read_csv(INFILE, index_col=0)

G = nx.DiGraph()

for source in m.index:
    for target in m.columns:
        weight = m.loc[source, target]
        if weight >= 0.3 and source != target:
            G.add_edge(source, target, weight=weight)

pos = nx.spring_layout(G, seed=7, k=1.1)

plt.figure(figsize=(6, 5))

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=1700,
    edgecolors="black",
    linewidths=1.2
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=10,
    font_weight="bold"
)

edges = G.edges(data=True)
widths = [2 + 5 * d["weight"] for _, _, d in edges]

nx.draw_networkx_edges(
    G,
    pos,
    arrowstyle="-|>",
    arrowsize=18,
    width=widths,
    connectionstyle="arc3,rad=0.12"
)

edge_labels = {
    (u, v): f"{d['weight']:.2f}"
    for u, v, d in edges
}

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=8
)

plt.title("Ambiguous-cell identity neighbourhood graph")
plt.axis("off")
plt.tight_layout()

plt.savefig(f"{OUTDIR}/identity_neighbourhood_graph.png", dpi=300)
plt.savefig(f"{OUTDIR}/identity_neighbourhood_graph.pdf")
plt.close()

print("Saved:")
print(f"{OUTDIR}/identity_neighbourhood_graph.png")
print(f"{OUTDIR}/identity_neighbourhood_graph.pdf")
