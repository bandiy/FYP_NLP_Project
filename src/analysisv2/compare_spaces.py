import numpy as np
import json
import random
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

#paths

PHON_KNN_PATH = Path("data/corpora/russian2/phonetic_knn_graph.json")
PHON_IPA_PATH = Path("data/corpora/russian2/lemma_ipa.tsv")

SEM_EMB_PATH = Path("data/corpora/russian2/semantic_embeddings.npy")
SEM_LEMMA_PATH = Path("data/corpora/russian2/semantic_lemmas.txt")

#K value
Ks = [1, 3, 5, 10, 15, 20]


#load

def load_phonetic():
    with open(PHON_KNN_PATH, encoding="utf-8") as f:
        knn = json.load(f)

    phon_lemmas = []
    with open(PHON_IPA_PATH, encoding="utf-8") as f:
        for line in f:
            lemma, _ = line.strip().split("\t")
            phon_lemmas.append(lemma)

    return knn, phon_lemmas


def load_semantic():
    embeddings = np.load(SEM_EMB_PATH)
    with open(SEM_LEMMA_PATH, encoding="utf-8") as f:
        lemmas = [line.strip() for line in f]
    return embeddings, lemmas


#sem knn

def build_semantic_knn(embeddings, lemmas, K):
    sim_matrix = cosine_similarity(embeddings)
    knn = {}

    for i, lemma in enumerate(lemmas):
        nearest = np.argsort(-sim_matrix[i])[1:K+1]
        knn[lemma] = [lemmas[j] for j in nearest]

    return knn


#main

def main():

    phon_knn, phon_lemmas = load_phonetic()
    sem_embeddings, sem_lemmas = load_semantic()

    intersection = list(set(phon_lemmas) & set(sem_lemmas))
    print("Intersection size:", len(intersection))

    for K in Ks:

        print(f"\n==============================")
        print(f"          K = {K}")
        print(f"==============================")

        sem_knn = build_semantic_knn(sem_embeddings, sem_lemmas, K)

        overlaps = []
        random_overlaps = []
        rank_corrs = []

        for word in intersection:

            phon_neighbors = [
                n["lemma"] for n in phon_knn[word][:K]
            ]
            sem_neighbors = sem_knn[word]

            phon_set = set(phon_neighbors)
            sem_set = set(sem_neighbors)

            # overlap
            shared = phon_set & sem_set
            overlaps.append(len(shared) / K)

            # random baseline
            random_neighbors = set(random.sample(intersection, K))
            random_overlaps.append(
                len(random_neighbors & sem_set) / K
            )

            # if k >=2 rank agreement
            if K >= 2:

                phon_rank = {w: i for i, w in enumerate(phon_neighbors)}
                sem_rank = {w: i for i, w in enumerate(sem_neighbors)}

                common = phon_set & sem_set

                if len(common) >= 2:
                    sem_ranks = [sem_rank[w] for w in common]
                    phon_ranks = [phon_rank[w] for w in common]

                    corr, _ = spearmanr(sem_ranks, phon_ranks)
                    if not np.isnan(corr):
                        rank_corrs.append(corr)

        #print

        print("Average overlap:", np.mean(overlaps))
        print("Random baseline:", np.mean(random_overlaps))
        print("Signal ratio:", np.mean(overlaps) / (np.mean(random_overlaps) + 1e-8))
        print("Percent with ≥1 overlap:",
              np.mean([o > 0 for o in overlaps]) * 100)

        if rank_corrs:
            print("Mean rank correlation:", np.mean(rank_corrs))
        else:
            print("Mean rank correlation: N/A")


if __name__ == "__main__":
    main()