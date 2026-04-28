import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

# config
JSON_PATH = "data/corpora/russian2/phonetic_knn_graph.json"
MIN_CLUSTER_SIZE = 5
SEED = 42


# laod
with open(JSON_PATH, "r", encoding="utf-8") as f:
    knn = json.load(f)

G = nx.Graph()

for word, neighbours in knn.items():
    for n in neighbours:
        G.add_edge(word, n["lemma"], weight=n["distance"])

print(f"Nodes: {len(G.nodes)}")
print(f"Edges: {len(G.edges)}")

# community detection
communities = list(community.greedy_modularity_communities(G))
communities = [c for c in communities if len(c) >= MIN_CLUSTER_SIZE]

print(f"Clusters retained: {len(communities)}")

pos = nx.spring_layout(G, k=0.25, seed=SEED)

#plotting
plt.figure(figsize=(15, 10))
ax = plt.gca()

# Slightly expand layout to reduce centre compression
pos = nx.spring_layout(G, k=0.32, seed=42)

# Draw very faint edges
nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.03,
    edge_color="black",
    width=0.3
)

colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))

for i, comm in enumerate(communities):

    nodes = list(comm)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_size=42,
        node_color=[colors[i]],
        edgecolors="white",
        linewidths=0.25,
        alpha=0.92
    )

    # Cluster centroid label
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    cx, cy = np.mean(xs), np.mean(ys)

    plt.text(
        cx,
        cy,
        str(i),
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        color="black"
    )

plt.title(
    "Phonetic Community Structure\n(Weighted Feature Edit Distance + KNN Graph)",
    fontsize=14,
    fontweight="bold",
    pad=15
)

plt.text(
    0.01,
    0.01,
    f"Nodes: {len(G.nodes)}   Edges: {len(G.edges)}   Clusters: {len(communities)}",
    transform=ax.transAxes,
    fontsize=10
)

plt.axis("equal")
plt.axis("off")
plt.tight_layout()
plt.show()