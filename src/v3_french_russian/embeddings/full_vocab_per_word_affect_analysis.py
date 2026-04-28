import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import panphon.distance

# conf
ALPHA = 0.90

BASE = Path("data/corpora/russian_french/processed")

RU_SEM_PATH = BASE / "russian_semantic_embeddings.npy"
FR_SEM_PATH = BASE / "french_semantic_embeddings.npy"

RU_LEMMA_PATH = BASE / "russian_lemma_frequencies.tsv"
FR_LEMMA_PATH = BASE / "french_lemma_frequencies.tsv"

RU_IPA_PATH = BASE / "russian_lemma_ipa.tsv"
FR_IPA_PATH = BASE / "french_lemma_ipa.tsv"

gold_pairs = {
"брат": "frère",
"мать": "mère",
"отец": "père",
"сын": "fils",
"зуб": "dent",
"нос": "nez",
"имя": "nom",
"ночь": "nuit",
"солнце": "soleil",
"земля": "terre",
"жить": "vivre",
"умереть": "mourir",
"видеть": "voir",
"дать": "donner",
"новый": "neuf",
"брать": "prendre",
"знать": "connaître"
}

# func
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

# main

def main():

    print("Loading data...")

    ru_sem = np.load(RU_SEM_PATH)
    fr_sem = np.load(FR_SEM_PATH)

    ru_lemmas = load_lemmas(RU_LEMMA_PATH)
    fr_lemmas = load_lemmas(FR_LEMMA_PATH)

    ru_ipa = load_ipa(RU_IPA_PATH)
    fr_ipa = load_ipa(FR_IPA_PATH)

    sem_matrix = cosine_similarity(ru_sem, fr_sem)
    dist = panphon.distance.Distance()

    results = []

    for ru_word, fr_word in gold_pairs.items():

        ru_idx = ru_lemmas.index(ru_word)
        fr_idx = fr_lemmas.index(fr_word)

        # semantic ranking
        sem_scores = sem_matrix[ru_idx]
        sem_ranked = np.argsort(-sem_scores)
        sem_rank = list(sem_ranked).index(fr_idx) + 1
        sem_rr = 1 / sem_rank

        # ;phonetic scores
        phon_scores = []
        for candidate in fr_lemmas:
            if candidate not in fr_ipa:
                phon_scores.append(0)
            else:
                phon_scores.append(
                    phonetic_similarity(dist, ru_ipa[ru_word], fr_ipa[candidate])
                )

        phon_scores = np.array(phon_scores)

        # hybrid fusion
        sem_norm = minmax(sem_scores)
        phon_norm = minmax(phon_scores)
        combined = ALPHA * sem_norm + (1 - ALPHA) * phon_norm

        ranked = np.argsort(-combined)
        hyb_rank = list(ranked).index(fr_idx) + 1
        hyb_rr = 1 / hyb_rank

        delta = hyb_rr - sem_rr

        results.append((ru_word, fr_word, sem_rank, hyb_rank, delta))

    # sort by improvement
    results.sort(key=lambda x: x[4], reverse=True)

    print("\n=== PER-WORD EFFECT (sorted by Δ RR) ===\n")

    for ru_word, fr_word, sem_rank, hyb_rank, delta in results:
        print(f"{ru_word} → {fr_word}")
        print(f"  Semantic rank: {sem_rank}")
        print(f"  Hybrid rank:   {hyb_rank}")
        print(f"  Δ RR: {delta:.4f}\n")

if __name__ == "__main__":
    main()