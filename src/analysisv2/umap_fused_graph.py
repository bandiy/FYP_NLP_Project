import numpy as np
import umap
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances

SEM_PATH = Path("data/corpora/russian2/semantic_embeddings.npy")
PHON_PATH = Path("data/corpora/russian2/phonetic_distance_matrix_6000.npy")
LEMMA_PATH = Path("data/corpora/russian2/semantic_lemmas.txt")
IPA_PATH = Path("data/corpora/russian2/lemma_ipa.tsv")

ALPHAS = [0, 0.3, 0.7, 1]


def normalize(matrix):
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())


def load_data():
    semantic = np.load(SEM_PATH)
    phonetic_full = np.load(PHON_PATH)

    with open(LEMMA_PATH, encoding="utf-8") as f:
        sem_lemmas = [l.strip() for l in f]

    ipa_lemmas = []
    with open(IPA_PATH, encoding="utf-8") as f:
        for line in f:
            lemma, _ = line.strip().split("\t")
            ipa_lemmas.append(lemma)

    index_map = {lemma: i for i, lemma in enumerate(ipa_lemmas)}
    phon_indices = [index_map[l] for l in sem_lemmas]

    phonetic = phonetic_full[np.ix_(phon_indices, phon_indices)]

    return semantic, phonetic


def build_umap(distance_matrix):
    reducer = umap.UMAP(
        metric="precomputed",
        n_neighbors=20,
        min_dist=0.1,
        random_state=42
    )
    return reducer.fit_transform(distance_matrix)


def main():
    semantic, phonetic = load_data()

    semantic_dist = cosine_distances(semantic)

    semantic_dist = normalize(semantic_dist)
    phonetic = normalize(phonetic)

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Alpha = {a}" for a in ALPHAS]
    )

    positions = [(1,1), (1,2), (2,1), (2,2)]

    for alpha, (r,c) in zip(ALPHAS, positions):

        fused = alpha * semantic_dist + (1 - alpha) * phonetic
        embedding = build_umap(fused)

        fig.add_trace(
            go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(size=3),
                showlegend=False
            ),
            row=r, col=c
        )

    fig.update_layout(
        height=900,
        width=900,
        title="Phonetic–Semantic Manifold Blending (Separate Projections)",
        template="plotly_white"
    )

    fig.show()
    
if __name__ == "__main__":
    main()
