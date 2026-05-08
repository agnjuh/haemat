import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

IN = "results/dynamics/identity_displacement.csv"
OUT = "results/dynamics/displacement_graph.png"

df = pd.read_csv(IN)

agg = df.groupby(["top1","top2"])["displacement"].mean().reset_index()

G = nx.DiGraph()

for _, row in agg.iterrows():
    if row["top1"] == row["top2"]:
        continue
    if row["displacement"] < 0.05:
        continue

    G.add_edge(
        row["top1"],
        row["top2"],
        weight=row["displacement"]
    )

pos = nx.spring_layout(G, seed=42)

weights = [G[u][v]['weight']*10 for u,v in G.edges()]

plt.figure(figsize=(6,6))
nx.draw(G, pos, with_labels=True, node_size=2000, width=weights)

edge_labels = {(u,v): f"{d['weight']:.2f}" for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Identity displacement graph")
plt.tight_layout()
plt.savefig(OUT, dpi=300)
plt.close()

print("Saved:", OUT)
