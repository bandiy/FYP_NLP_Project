import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import panphon.distance

ALPHA = 0.90

BASE = Path("data/corpora/russian_french/processed")

RU_SEM_PATH = BASE / "russian_semantic_embeddings.npy"
FR_SEM_PATH = BASE / "french_semantic_embeddings.npy"

RU_LEMMA_PATH = BASE / "russian_lemma_frequencies.tsv"
FR_LEMMA_PATH = BASE / "french_lemma_frequencies.tsv"

RU_IPA_PATH = BASE / "russian_lemma_ipa.tsv"
FR_IPA_PATH = BASE / "french_lemma_ipa.tsv"

TARGETS = ["земля", "новый"]

def load_lemmas(path):
    with path.open(encoding="utf-8") as f:
        return [line.strip().split("\t")[0] for line in f]

def load_ipa(path):
    ipa_map = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            lemma, ipa = line.strip().split("\t")
            ipa_map[lemma] = ipa
    return ipa_map

def phonetic_similarity(dist, ipa1, ipa2):
    raw = dist.weighted_feature_edit_distance(ipa1, ipa2)
    norm = raw / max(len(ipa1), len(ipa2), 1)
    return 1 / (1 + norm)

def minmax(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def main():

    print("Loading data...\n")

    ru_sem = np.load(RU_SEM_PATH)
    fr_sem = np.load(FR_SEM_PATH)

    ru_lemmas = load_lemmas(RU_LEMMA_PATH)
    fr_lemmas = load_lemmas(FR_LEMMA_PATH)

    ru_ipa = load_ipa(RU_IPA_PATH)
    fr_ipa = load_ipa(FR_IPA_PATH)

    sem_matrix = cosine_similarity(ru_sem, fr_sem)
    dist = panphon.distance.Distance()

    for ru_word in TARGETS:

        print("=" * 60)
        print(f"RUSSIAN WORD: {ru_word}\n")

        ru_idx = ru_lemmas.index(ru_word)
        sem_scores = sem_matrix[ru_idx]

        sem_ranked = np.argsort(-sem_scores)
        top5 = sem_ranked[:5]

        print("Top 5 semantic candidates:\n")

        for rank, idx in enumerate(top5, start=1):
            fr_word = fr_lemmas[idx]
            sem_score = sem_scores[idx]

            if fr_word in fr_ipa:
                phon_score = phonetic_similarity(
                    dist, ru_ipa[ru_word], fr_ipa[fr_word]
                )
            else:
                phon_score = 0.0

            print(f"{rank}. {fr_word}")
            print(f"   Semantic score: {sem_score:.4f}")
            print(f"   Phonetic sim:   {phon_score:.4f}\n")

        # Hybrid ranking for comparison
        phon_scores = []
        for candidate in fr_lemmas:
            if candidate not in fr_ipa:
                phon_scores.append(0)
            else:
                phon_scores.append(
                    phonetic_similarity(dist, ru_ipa[ru_word], fr_ipa[candidate])
                )

        sem_norm = minmax(sem_scores)
        phon_norm = minmax(phon_scores)
        combined = ALPHA * sem_norm + (1 - ALPHA) * phon_norm

        hybrid_ranked = np.argsort(-combined)
        hybrid_top3 = hybrid_ranked[:3]

        print("Top 3 hybrid candidates:\n")
        for rank, idx in enumerate(hybrid_top3, start=1):
            print(f"{rank}. {fr_lemmas[idx]}")
        print("\n")

if __name__ == "__main__":
    main()