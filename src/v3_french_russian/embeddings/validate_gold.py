import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import panphon.distance

#paths

BASE = Path("data/corpora/russian_french/processed")

RU_SEM_PATH = BASE / "russian_semantic_embeddings.npy"
FR_SEM_PATH = BASE / "french_semantic_embeddings.npy"

RU_LEMMA_PATH = BASE / "russian_lemma_frequencies.tsv"
FR_LEMMA_PATH = BASE / "french_lemma_frequencies.tsv"

RU_IPA_PATH = BASE / "russian_lemma_ipa.tsv"
FR_IPA_PATH = BASE / "french_lemma_ipa.tsv"

# goldset

gold_pairs = {

# Kinship
"брат": "frère",
"мать": "mère",
"отец": "père",
"сын": "fils",

# Body
"зуб": "dent",
"нос": "nez",
"нога": "pied",

# Nature
"солнце": "soleil",
"луна": "lune",
"земля": "terre",
"огонь": "feu",

# Core verbs
"дать": "donner",
"брать": "prendre",
"знать": "connaître",
"видеть": "voir",
"жить": "vivre",
"умереть": "mourir",

# Adjectives
"новый": "neuf",

# Core nouns
"имя": "nom",
"ночь": "nuit",
"день": "jour"
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

    print("\n=== GOLD VALIDATION REPORT ===\n")

    for ru_word, fr_word in gold_pairs.items():

        print(f"{ru_word} → {fr_word}")

        # Presence checks
        if ru_word not in ru_lemmas:
            print("RU word not in corpus")
            continue

        if fr_word not in fr_lemmas:
            print("FR word not in corpus")
            continue

        if ru_word not in ru_ipa:
            print("RU word missing IPA")
            continue

        if fr_word not in fr_ipa:
            print("FR word missing IPA")
            continue

        ru_idx = ru_lemmas.index(ru_word)
        fr_idx = fr_lemmas.index(fr_word)

        # Semantic rank
        sem_scores = sem_matrix[ru_idx]
        sem_ranked = np.argsort(-sem_scores)
        sem_rank = list(sem_ranked).index(fr_idx) + 1

        # Phonetic rank
        phon_scores = []
        for candidate in fr_lemmas:
            if candidate not in fr_ipa:
                phon_scores.append(0)
            else:
                phon_scores.append(
                    phonetic_similarity(dist, ru_ipa[ru_word], fr_ipa[candidate])
                )

        phon_scores = np.array(phon_scores)
        phon_ranked = np.argsort(-phon_scores)
        phon_rank = list(phon_ranked).index(fr_idx) + 1

        print(f"  Semantic rank: {sem_rank}")
        print(f"  Phonetic rank: {phon_rank}")

        # Flag extreme outliers
        if sem_rank > 1000:
            print("Very low semantic alignment")

        if phon_rank > 3000:
            print("Very low phonetic alignment")

        print()

if __name__ == "__main__":
    main()