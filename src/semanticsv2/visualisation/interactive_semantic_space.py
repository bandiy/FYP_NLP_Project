import numpy as np
import json
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_distances
from pathlib import Path


#plaths

UMAP_PATH = Path("data/corpora/russian2/semantic_umap.npy")
EMB_PATH = Path("data/corpora/russian2/semantic_embeddings.npy")
LEMMA_PATH = Path("data/corpora/russian2/semantic_lemmas.txt")
CLUSTER_PATH = Path("data/corpora/russian2/semantic_clusters.npy")
TRANSLATION_PATH = Path("data/corpora/russian2/lemma_translations.json")

TOP_K = 5


#hover

def compute_hover_text(embeddings, lemmas, translations, clusters):

    dists_matrix = cosine_distances(embeddings)
    hover_text = []

    for i, lemma in enumerate(lemmas):

        english = translations.get(lemma, "")
        cluster_id = clusters[i]

        nearest_ids = np.argsort(dists_matrix[i])[1:TOP_K+1]

        neighbour_lines = []

        for j in nearest_ids:
            ru = lemmas[j]
            en = translations.get(ru, "")
            neighbour_lines.append(f"{ru} ({en})")

        neighbour_str = "<br>".join(neighbour_lines)

        hover_text.append(
            f"<span style='font-size:18px'><b>{lemma}</b></span><br>"
            f"<span style='color:gray'>{english}</span><br>"
            f"Cluster: {cluster_id}<br><br>"
            f"<b>Nearest neighbours:</b><br>{neighbour_str}"
        )

    return hover_text


#main

def main():

    embedding_2d = np.load(UMAP_PATH)
    embeddings = np.load(EMB_PATH)
    clusters = np.load(CLUSTER_PATH)

    with open(LEMMA_PATH, encoding="utf-8") as f:
        lemmas = [line.strip() for line in f]

    with open(TRANSLATION_PATH, encoding="utf-8") as f:
        translations = json.load(f)

    hover_text = compute_hover_text(
        embeddings,
        lemmas,
        translations,
        clusters
    )

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        mode="markers",
        marker=dict(
            size=6,
            color=clusters,
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(
                title="Cluster ID"
            )
        ),
        text=hover_text,
        hoverinfo="text"
    ))

    fig.update_layout(
        title="Interactive Semantic Space Explorer (Dostoyevsky Lemmas)",
        template="plotly_white",
        height=850,
        hoverlabel=dict(
            bgcolor="white",
            font_size=13
        )
    )

    fig.show()

    fig.write_html(
        "semantic_space_explorer.html",
        include_plotlyjs=True,
        full_html=True
    )

    print("Saved as semantic_space_explorer.html")


if __name__ == "__main__":
    main()
