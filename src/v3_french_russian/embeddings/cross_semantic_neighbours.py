import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

RU_EMB_PATH = Path("data/corpora/russian_french/processed/russian_semantic_embeddings.npy")
FR_EMB_PATH = Path("data/corpora/russian_french/processed/french_semantic_embeddings.npy")

RU_LEMMA_PATH = Path("data/corpora/russian_french/processed/russian_lemma_frequencies.tsv")
FR_LEMMA_PATH = Path("data/corpora/russian_french/processed/french_lemma_frequencies.tsv")

TOP_K = 10

def load_lemmas(path):
    with path.open(encoding="utf-8") as f:
        return [line.strip().split("\t")[0] for line in f]

def main():
    ru_emb = np.load(RU_EMB_PATH)
    fr_emb = np.load(FR_EMB_PATH)

    ru_lemmas = load_lemmas(RU_LEMMA_PATH)
    fr_lemmas = load_lemmas(FR_LEMMA_PATH)

    sim_matrix = cosine_similarity(ru_emb, fr_emb)

    print("\nRussian test:\n")

    test_words = ["брат", "мать", "ночь", "новый", "имя"]

    for word in test_words:
        if word not in ru_lemmas:
            print(f"{word} not found in RU vocab.")
            continue

        idx = ru_lemmas.index(word)
        sims = sim_matrix[idx]

        top_indices = np.argsort(-sims)[:TOP_K]

        print(f"\nRussian word: {word}")
        for i in top_indices:
            print(f"  {fr_lemmas[i]}  ({sims[i]:.4f})")

if __name__ == "__main__":
    main()
