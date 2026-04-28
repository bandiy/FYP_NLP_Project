import numpy as np
from pathlib import Path
import json

# config
MATRIX_PATH = "data/corpora/russian2/phonetic_distance_matrix_6000.npy" # check 3000 or 6000
IPA_PATH = "data/corpora/russian2/lemma_ipa.tsv"
OUT_PATH = "data/corpora/russian2/phonetic_knn_graph.json"

K = 6 #10


# load
def load_lemmas():
    lemmas = []

    with open(IPA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            lemma, ipa = line.strip().split("\t")
            lemmas.append(lemma)

    return lemmas


# build KNN graph
def build_knn():
    print("Loading matrix...")
    matrix = np.load(MATRIX_PATH)

    print("Loading lemmas...")
    lemmas = load_lemmas()

    n = len(lemmas)
    knn_graph = {}

    print("Building KNN graph...")

    for i in range(n):
        distances = matrix[i]

        # exclude self
        nearest_idx = np.argsort(distances)[1:K+1]

        neighbours = [
            {
                "lemma": lemmas[j],
                "distance": float(distances[j])
            }
            for j in nearest_idx
        ]

        knn_graph[lemmas[i]] = neighbours

        if i % 100 == 0:
            print(f"{i}/{n}")

    print("Saving...")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(knn_graph, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    build_knn()
