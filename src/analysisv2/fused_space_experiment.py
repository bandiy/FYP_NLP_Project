import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances

# Paths
SEM_PATH = Path("data/corpora/russian2/semantic_embeddings.npy")
PHON_PATH = Path("data/corpora/russian2/phonetic_distance_matrix_6000.npy")
LEMMA_PATH = Path("data/corpora/russian2/semantic_lemmas.txt")
IPA_PATH = Path("data/corpora/russian2/lemma_ipa.tsv")

K = 5
ALPHAS = [0, 0.25, 0.5, 0.75, 1]


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


def compute_knn_overlap(dist_matrix_1, dist_matrix_2):
    n = len(dist_matrix_1)
    overlaps = []

    for i in range(n):
        nn1 = set(np.argsort(dist_matrix_1[i])[1:K+1])
        nn2 = set(np.argsort(dist_matrix_2[i])[1:K+1])
        overlaps.append(len(nn1 & nn2) / K)

    return np.mean(overlaps)


def compute_global_corr(dist_matrix_1, dist_matrix_2):
    triu = np.triu_indices(len(dist_matrix_1), k=1)
    flat1 = dist_matrix_1[triu]
    flat2 = dist_matrix_2[triu]
    corr, _ = spearmanr(flat1, flat2)
    return corr


def main():
    semantic, phonetic = load_intersection()

    semantic_dist = cosine_distances(semantic)

    phonetic = normalize(phonetic)
    semantic_dist = normalize(semantic_dist)


    for alpha in ALPHAS:
        fused = alpha * semantic_dist + (1 - alpha) * phonetic

        overlap = compute_knn_overlap(fused, semantic_dist)
        corr = compute_global_corr(fused, semantic_dist)

        print(f"\nAlpha = {alpha}")
        print("Neighbour overlap vs semantic:", overlap)
        print("Global correlation vs semantic:", corr)


if __name__ == "__main__":
    main()
