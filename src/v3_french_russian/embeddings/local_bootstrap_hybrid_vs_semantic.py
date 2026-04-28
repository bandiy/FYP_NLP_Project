import numpy as np
import random
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import panphon.distance


# =====================
# CONFIG
# =====================

ALPHA = 0.60
BOOTSTRAP_ITERATIONS = 500


RU_EMB_PATH = Path("data/corpora/russian_french/processed/russian_semantic_embeddings.npy")
FR_EMB_PATH = Path("data/corpora/russian_french/processed/french_semantic_embeddings.npy")

RU_LEMMA_PATH = Path("data/corpora/russian_french/processed/russian_lemma_frequencies.tsv")
FR_LEMMA_PATH = Path("data/corpora/russian_french/processed/french_lemma_frequencies.tsv")

RU_IPA_PATH = Path("data/corpora/russian_french/processed/russian_lemma_ipa.tsv")
FR_IPA_PATH = Path("data/corpora/russian_french/processed/french_lemma_ipa.tsv")


gold_pairs = {

"брат": "frère",
"мать": "mère",
"отец": "père",
"сын": "fils",
"сестра": "sœur",
"два": "deux",
"три": "trois",
"четыре": "quatre",
"пять": "cinq",
"шесть": "six",
"семь": "sept",
"восемь": "huit",
"девять": "neuf",
"десять": "dix",
"новый": "neuf",
"имя": "nom",
"ночь": "nuit",
"нос": "nez",
"зуб": "dent",
"сто": "cent",
"сердце": "cœur",
"огонь": "feu",
"земля": "terre",
"знать": "connaître",
"видеть": "voir",
"нести": "porter",
"стоять": "être",
"сидеть": "asseoir",
"дать": "donner",
"брать": "prendre",
"жить": "vivre",
"умереть": "mourir",
"ветер": "vent",
"звезда": "étoile",
"долгий": "long",
"вести": "voie",
"кровь": "cru"
}


# =====================
# HELPERS
# =====================

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

def minmax(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def phonetic_similarity(dist, ipa1, ipa2):
    raw = dist.weighted_feature_edit_distance(ipa1, ipa2)
    norm = raw / max(len(ipa1), len(ipa2), 1)
    return 1 / (1 + norm)


# =====================
# MAIN
# =====================

def main():

    print("Precomputing ranks...")

    ru_emb = np.load(RU_EMB_PATH)
    fr_emb = np.load(FR_EMB_PATH)

    ru_lemmas = load_lemmas(RU_LEMMA_PATH)
    fr_lemmas = load_lemmas(FR_LEMMA_PATH)

    ru_ipa = load_ipa(RU_IPA_PATH)
    fr_ipa = load_ipa(FR_IPA_PATH)

    sim_matrix = cosine_similarity(ru_emb, fr_emb)
    dist = panphon.distance.Distance()

    semantic_rr = []
    hybrid_rr = []

    for ru_word, fr_gold in gold_pairs.items():

        if ru_word not in ru_lemmas or fr_gold not in fr_lemmas:
            continue
        if ru_word not in ru_ipa or fr_gold not in fr_ipa:
            continue

        ru_idx = ru_lemmas.index(ru_word)
        gold_idx = fr_lemmas.index(fr_gold)

        sem_scores = sim_matrix[ru_idx]
        sem_ranked = np.argsort(-sem_scores)

        # Semantic RR
        sem_rank = list(sem_ranked).index(gold_idx) + 1
        semantic_rr.append(1 / sem_rank)

        # Hybrid shortlist
        candidates = []
        for i in sem_ranked[:200]:
            fr_word = fr_lemmas[i]
            if fr_word not in fr_ipa:
                continue

            sem_score = sem_scores[i]
            phon_sim = phonetic_similarity(dist, ru_ipa[ru_word], fr_ipa[fr_word])
            candidates.append((i, sem_score, phon_sim))

        idxs, sem_vals, phon_vals = zip(*candidates)

        sem_norm = minmax(sem_vals)
        phon_norm = minmax(phon_vals)

        combined = ALPHA * sem_norm + (1 - ALPHA) * phon_norm
        hybrid_ranked = [idxs[i] for i in np.argsort(-combined)]

        if gold_idx in hybrid_ranked:
            hyb_rank = hybrid_ranked.index(gold_idx) + 1
        else:
            hyb_rank = len(hybrid_ranked) + 1

        hybrid_rr.append(1 / hyb_rank)

    semantic_rr = np.array(semantic_rr)
    hybrid_rr = np.array(hybrid_rr)

    print("Bootstrap running...")

    improvements = []

    n = len(semantic_rr)

    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = np.random.choice(n, size=n, replace=True)
        sem_mrr = np.mean(semantic_rr[indices])
        hyb_mrr = np.mean(hybrid_rr[indices])
        improvements.append(hyb_mrr - sem_mrr)

    improvements = np.array(improvements)

    print("\n=== BOOTSTRAP RESULTS ===")
    print(f"Mean improvement: {np.mean(improvements):.5f}")
    print(f"Std deviation: {np.std(improvements):.5f}")
    print(f"95% CI: [{np.percentile(improvements,2.5):.5f}, {np.percentile(improvements,97.5):.5f}]")
    print(f"% Hybrid > Semantic: {np.mean(improvements > 0)*100:.2f}%")
    print(f"Iterations: {BOOTSTRAP_ITERATIONS}")


if __name__ == "__main__":
    main()