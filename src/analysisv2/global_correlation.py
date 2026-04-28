import numpy as np
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import spearmanr, pearsonr

#paths

PHON_MATRIX_PATH = Path("data/corpora/russian2/phonetic_distance_matrix_6000.npy")
PHON_IPA_PATH = Path("data/corpora/russian2/lemma_ipa.tsv")

SEM_EMB_PATH = Path("data/corpora/russian2/semantic_embeddings.npy")
SEM_LEMMA_PATH = Path("data/corpora/russian2/semantic_lemmas.txt")

def load_phonetic():

    matrix = np.load(PHON_MATRIX_PATH)

    lemmas = []
    with open(PHON_IPA_PATH, encoding="utf-8") as f:
        for line in f:
            lemma, ipa = line.strip().split("\t")
            lemmas.append(lemma)

    return matrix, lemmas


def load_semantic():

    embeddings = np.load(SEM_EMB_PATH)

    with open(SEM_LEMMA_PATH, encoding="utf-8") as f:
        lemmas = [line.strip() for line in f]

    return embeddings, lemmas

#main
# split where phonetic k=1 and semantic k=1 are the same and see spearman

def main():
    phon_matrix, phon_lemmas = load_phonetic()

    sem_embeddings, sem_lemmas = load_semantic()

    intersection = sorted(list(set(phon_lemmas) & set(sem_lemmas)))
    print("Intersection size:", len(intersection))

    # index maps
    phon_index = {lemma: i for i, lemma in enumerate(phon_lemmas)}
    sem_index = {lemma: i for i, lemma in enumerate(sem_lemmas)}

    # extracct phon matrix
    indices = [phon_index[w] for w in intersection]
    phon_sub = phon_matrix[np.ix_(indices, indices)]

    # compute sem matrix
    sem_indices = [sem_index[w] for w in intersection]
    sem_sub_embeddings = sem_embeddings[sem_indices]
    sem_sub = cosine_distances(sem_sub_embeddings)

    # flatten triangle
    triu_indices = np.triu_indices(len(intersection), k=1)

    phon_flat = phon_sub[triu_indices]
    sem_flat = sem_sub[triu_indices]

    print("Number of pairwise comparisons:", len(phon_flat))

    # correlation
    print("\n--- GLOBAL CORRELATION ---")

    spearman_corr, spearman_p = spearmanr(phon_flat, sem_flat)
    pearson_corr, pearson_p = pearsonr(phon_flat, sem_flat)

    print("Spearman correlation:", spearman_corr)

    print("Pearson correlation:", pearson_corr)


if __name__ == "__main__":
    main()
