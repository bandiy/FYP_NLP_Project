from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

RU_PATH = Path("data/corpora/russian_french/processed/russian_lemma_frequencies.tsv")
FR_PATH = Path("data/corpora/russian_french/processed/french_lemma_frequencies.tsv")

OUT_RU = Path("data/corpora/russian_french/processed/russian_semantic_embeddings.npy")
OUT_FR = Path("data/corpora/russian_french/processed/french_semantic_embeddings.npy")

def load_lemmas(path):
    with path.open(encoding="utf-8") as f:
        return [line.strip().split("\t")[0] for line in f]

def main():
    ru_lemmas = load_lemmas(RU_PATH)
    fr_lemmas = load_lemmas(FR_PATH)

    all_words = ru_lemmas + fr_lemmas

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    embeddings = model.encode(
        all_words,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    ru_emb = embeddings[:len(ru_lemmas)]
    fr_emb = embeddings[len(ru_lemmas):]

    np.save(OUT_RU, ru_emb)
    np.save(OUT_FR, fr_emb)

    print("Saved Russian embeddings:", OUT_RU)
    print("Saved French embeddings:", OUT_FR)

if __name__ == "__main__":
    main()
