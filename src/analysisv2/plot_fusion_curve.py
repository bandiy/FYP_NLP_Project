import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances
import plotly.graph_objects as go


SEM_PATH = Path("data/corpora/russian2/semantic_embeddings.npy")
PHON_PATH = Path("data/corpora/russian2/phonetic_distance_matrix_6000.npy")
LEMMA_PATH = Path("data/corpora/russian2/semantic_lemmas.txt")
IPA_PATH = Path("data/corpora/russian2/lemma_ipa.tsv")

K = 5
ALPHAS = np.linspace(0, 1, 11)


def normalize(matrix):
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())


def load_intersection():
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


def compute_knn_overlap(dist1, dist2):
    n = len(dist1)
    overlaps = []

    for i in range(n):
        nn1 = set(np.argsort(dist1[i])[1:K+1])
        nn2 = set(np.argsort(dist2[i])[1:K+1])
        overlaps.append(len(nn1 & nn2) / K)

    return np.mean(overlaps)


def compute_global_corr(dist1, dist2):
    triu = np.triu_indices(len(dist1), k=1)
    flat1 = dist1[triu]
    flat2 = dist2[triu]
    corr, _ = spearmanr(flat1, flat2)
    return corr

def main():
    semantic, phonetic = load_intersection()

    semantic_dist = cosine_distances(semantic)

    semantic_dist = normalize(semantic_dist)
    phonetic = normalize(phonetic)

    overlaps = []
    corrs = []

    for alpha in ALPHAS:
        fused = alpha * semantic_dist + (1 - alpha) * phonetic

        overlap = compute_knn_overlap(fused, semantic_dist)
        corr = compute_global_corr(fused, semantic_dist)

        overlaps.append(overlap)
        corrs.append(corr)

        print(f"Alpha {alpha:.2f} → Overlap {overlap:.3f} | Corr {corr:.3f}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ALPHAS,
        y=overlaps,
        mode="lines+markers",
        name="Neighbour Overlap",
        line=dict(width=3),
        marker=dict(size=7)
    ))

    fig.add_trace(go.Scatter(
        x=ALPHAS,
        y=corrs,
        mode="lines+markers",
        name="Global Correlation",
        line=dict(width=3),
        marker=dict(size=7)
    ))

    fig.update_layout(
        template="plotly_white",
        title="Fusion Curve: Phonetic–Semantic Interaction",
        xaxis_title="Alpha (Semantic Weight)",
        yaxis_title="Metric Value",
        height=850,
        hoverlabel=dict(
            bgcolor="white",
            font_size=13
        ),
        margin=dict(l=60, r=40, t=70, b=60)
    )

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    fig.write_html(
        "fusion_curve.html",
        include_plotlyjs=True,
        full_html=True
    )

    fig.show()

    print("Saved as fusion_curve.html")


if __name__ == "__main__":
    main()
