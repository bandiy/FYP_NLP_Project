import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import panphon.distance


# CONFIG
TOP_K = 5
ALPHA_VALUES = np.linspace(0, 1, 21)  # finer sweep


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


def main():

    ru_emb = np.load(RU_EMB_PATH)
    fr_emb = np.load(FR_EMB_PATH)

    ru_lemmas = load_lemmas(RU_LEMMA_PATH)
    fr_lemmas = load_lemmas(FR_LEMMA_PATH)

    ru_ipa = load_ipa(RU_IPA_PATH)
    fr_ipa = load_ipa(FR_IPA_PATH)

    sim_matrix = cosine_similarity(ru_emb, fr_emb)
    dist = panphon.distance.Distance()

    best_alpha = None
    best_mrr = -1

    print("\n=== ALPHA SWEEP ===\n")

    for ALPHA in ALPHA_VALUES:

        hybrid_mrr = 0
        total = 0

        for ru_word, fr_gold in gold_pairs.items():

            if ru_word not in ru_lemmas or fr_gold not in fr_lemmas:
                continue
            if ru_word not in ru_ipa or fr_gold not in fr_ipa:
                continue

            total += 1

            ru_idx = ru_lemmas.index(ru_word)
            gold_idx = fr_lemmas.index(fr_gold)

            sem_scores = sim_matrix[ru_idx]
            sem_ranked = np.argsort(-sem_scores)

            candidates = []
            for i in sem_ranked[:200]:
                fr_word = fr_lemmas[i]
                if fr_word not in fr_ipa:
                    continue

                sem_score = sem_scores[i]
                phon_sim = phonetic_similarity(
                    dist,
                    ru_ipa[ru_word],
                    fr_ipa[fr_word]
                )

                candidates.append((i, sem_score, phon_sim))

            if not candidates:
                continue

            idxs, sem_vals, phon_vals = zip(*candidates)

            sem_norm = minmax(sem_vals)
            phon_norm = minmax(phon_vals)

            combined = ALPHA * sem_norm + (1 - ALPHA) * phon_norm
            hybrid_ranked = [idxs[i] for i in np.argsort(-combined)]

            if gold_idx in hybrid_ranked:
                rank = hybrid_ranked.index(gold_idx) + 1
            else:
                rank = len(hybrid_ranked) + 1

            hybrid_mrr += 1 / rank

        hybrid_mrr /= total

        print(f"Alpha {ALPHA:.2f} | Hybrid MRR: {hybrid_mrr:.4f}")

        if hybrid_mrr > best_mrr:
            best_mrr = hybrid_mrr
            best_alpha = ALPHA

    print("\n=== BEST ALPHA ===")
    print(f"Best alpha: {best_alpha:.2f}")
    print(f"Best MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()