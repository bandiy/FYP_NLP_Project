import json
import numpy as np
import umap
import networkx as nx
import plotly.graph_objects as go
from networkx.algorithms import community

# config
MATRIX_PATH = "data/corpora/russian2/phonetic_distance_matrix_6000.npy"
JSON_PATH = "data/corpora/russian2/phonetic_knn_graph.json"
IPA_PATH = "data/corpora/russian2/lemma_ipa.tsv"

TOP_K = 5 #hovering tooltiop

# load data
matrix = np.load(MATRIX_PATH)

lemmas = []
ipas = []

with open(IPA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        lemma, ipa = line.strip().split("\t")
        lemmas.append(lemma)
        ipas.append(ipa)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    knn = json.load(f)

# umap
reducer = umap.UMAP(metric="precomputed", random_state=42)
embedding = reducer.fit_transform(matrix)

# community detection
G = nx.Graph()
for word, neighbours in knn.items():
    for n in neighbours:
        G.add_edge(word, n["lemma"])

communities = list(community.greedy_modularity_communities(G))

cluster_map = {}
for i, comm in enumerate(communities):
    for word in comm:
        cluster_map[word] = i

# build hover text
hover_text = []

lemma_index = {lemma: i for i, lemma in enumerate(lemmas)}

for i, lemma in enumerate(lemmas):

    ipa = ipas[i]
    cluster_id = cluster_map.get(lemma, "N/A")

    neighbours = knn.get(lemma, [])[:TOP_K]

    neighbour_lines = []
    for n in neighbours:
        neighbour_lemma = n["lemma"]
        idx = lemma_index[neighbour_lemma]
        neighbour_ipa = ipas[idx]

        neighbour_lines.append(
            f"{neighbour_ipa} ({neighbour_lemma}) — {n['distance']:.2f}"
        )

    neighbour_str = "<br>".join(neighbour_lines)

    hover_text.append(
        f"<span style='font-size:18px'><b>{ipa}</b></span><br>"
        f"<span style='color:gray'>{lemma}</span><br>"
        f"Cluster: {cluster_id}<br><br>"
        f"<b>Nearest neighbours:</b><br>{neighbour_str}"
    )

# plot
fig = go.Figure()

fig.add_trace(go.Scattergl(
    x=embedding[:, 0],
    y=embedding[:, 1],
    mode="markers",
    marker=dict(
        size=6,
        color=[cluster_map.get(l, 0) for l in lemmas],
        colorscale="Turbo",
        showscale=True,
        colorbar=dict(title="Cluster ID")
    ),
    text=hover_text,
    hoverinfo="text"
))

fig.update_layout(
    title="Interactive Phonetic Space Explorer (IPA View)",
    template="plotly_white",
    height=850,
    hoverlabel=dict(
        bgcolor="white",
        font_size=13
    )
)

fig.show()

# standalone
fig.write_html(
    "phonetic_space_explorer.html",
    include_plotlyjs=True, 
    full_html=True
)

print("Saved as phonetic_space_explorer.html")
