import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances
from pathlib import Path
import random

# paths
SEM_PATH = Path("data/corpora/russian2/semantic_embeddings.npy")
PHON_PATH = Path("data/corpora/russian2/phonetic_distance_matrix_6000.npy")
LEMMA_PATH = Path("data/corpora/russian2/semantic_lemmas.txt")
IPA_PATH = Path("data/corpora/russian2/lemma_ipa.tsv")

N_PERMUTATIONS = 200
BOOTSTRAP_SAMPLES = 100
BOOTSTRAP_SIZE = 1000


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

    # intersection indices
    index_map = {lemma: i for i, lemma in enumerate(ipa_lemmas)}
    phon_indices = [index_map[l] for l in sem_lemmas]

    phonetic = phonetic_full[np.ix_(phon_indices, phon_indices)]

    return semantic, phonetic


def compute_spearman(semantic, phonetic):
    sem_dist = cosine_distances(semantic)

    # flatten upper triangle only
    triu = np.triu_indices(len(semantic), k=1)

    sem_flat = sem_dist[triu]
    phon_flat = phonetic[triu]

    corr, p = spearmanr(sem_flat, phon_flat)
    return corr


def permutation_test(semantic, phonetic):
    print("Running permutation test...")
    real_corr = compute_spearman(semantic, phonetic)

    null_corrs = []

    for i in range(N_PERMUTATIONS):
        shuffled = np.random.permutation(semantic)
        corr = compute_spearman(shuffled, phonetic)
        null_corrs.append(corr)

        if i % 20 == 0:
            print(f"Permutation {i}/{N_PERMUTATIONS}")

    null_corrs = np.array(null_corrs)

    print("\n--- PERMUTATION RESULTS ---")
    print("Real correlation:", real_corr)
    print("Null mean:", null_corrs.mean())
    print("Null std:", null_corrs.std())

    return real_corr, null_corrs


def bootstrap_ci(semantic, phonetic):
    print("\nRunning bootstrap confidence interval...")
    corrs = []

    n = len(semantic)

    for i in range(BOOTSTRAP_SAMPLES):
        idx = np.random.choice(n, BOOTSTRAP_SIZE, replace=True)
        sem_sample = semantic[idx]
        phon_sample = phonetic[np.ix_(idx, idx)]

        corr = compute_spearman(sem_sample, phon_sample)
        corrs.append(corr)

        if i % 10 == 0:
            print(f"Bootstrap {i}/{BOOTSTRAP_SAMPLES}")

    corrs = np.array(corrs)

    lower = np.percentile(corrs, 2.5)
    upper = np.percentile(corrs, 97.5)

    print("\n--- BOOTSTRAP CI ---")
    print("Mean:", corrs.mean())
    print("95% CI:", lower, "to", upper)


def main():
    semantic, phonetic = load_data()

    real_corr, null_corrs = permutation_test(semantic, phonetic)
    bootstrap_ci(semantic, phonetic)


if __name__ == "__main__":
    main()
