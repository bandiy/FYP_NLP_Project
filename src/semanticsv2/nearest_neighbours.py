import numpy as np
import json
from sklearn.metrics.pairwise import cosine_distances
from pathlib import Path

EMB_PATH = Path("data/corpora/russian2/semantic_embeddings.npy")
LEMMA_PATH = Path("data/corpora/russian2/semantic_lemmas.txt")
TRANSLATION_PATH = Path("data/corpora/russian2/lemma_translations.json")

#embeddings
embeddings = np.load(EMB_PATH)

#lemmas
with open(LEMMA_PATH, encoding="utf-8") as f:
    lemmas = [line.strip() for line in f]

#translations
with open(TRANSLATION_PATH, encoding="utf-8") as f:
    translations = json.load(f)


def nearest(word, top_k=5):
    if word not in lemmas:
        print("Word not found.")
        return
    
    idx = lemmas.index(word)
    dists = cosine_distances([embeddings[idx]], embeddings)[0]
    nearest_ids = np.argsort(dists)[1:top_k+1]
    
    word_en = translations.get(word, "")

    print(f"\nNearest neighbours for: {word} ({word_en})\n")
    
    for i in nearest_ids:
        ru = lemmas[i]
        en = translations.get(ru, "")
        print(f"{ru} ({en})  ({dists[i]:.4f})")


if __name__ == "__main__":
    while True:
        w = input("\nEnter a lemma (or 'exit'): ")
        if w == "exit":
            break
        nearest(w)
