import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("data/corpora/russian2/lemma_frequencies.tsv")
OUT_EMB_PATH = Path("data/corpora/russian2/semantic_embeddings.npy")
OUT_LEMMA_PATH = Path("data/corpora/russian2/semantic_lemmas.txt")


def load_lemmas(min_freq=30):
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["lemma", "freq"])
    df = df[df["freq"] >= min_freq]
    return df["lemma"].tolist()


def main():
    print("Loading lemmas...")
    lemmas = load_lemmas()
    print(f"{len(lemmas)} lemmas loaded.")

    print("Loading semantic model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Encoding embeddings...")
    embeddings = model.encode(
        lemmas,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )


    print("Embeddings shape:", embeddings.shape)

    np.save(OUT_EMB_PATH, embeddings)

    with open(OUT_LEMMA_PATH, "w", encoding="utf-8") as f:
        for lemma in lemmas:
            f.write(lemma + "\n")

    print("Embeddings saved successfully.")


if __name__ == "__main__":
    main()
