# src/semanticsv1/cluster_embeddings.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path

EMB_PATH = Path("data/corpora/russian2/semantic_embeddings.npy")
OUT_CLUSTER_PATH = Path("data/corpora/russian2/semantic_clusters.npy")
OUT_PCA_PATH = Path("data/corpora/russian2/semantic_pca.npy")

N_CLUSTERS = 15


def main():
    embeddings = np.load(EMB_PATH)
    print("Embeddings shape:", embeddings.shape)

#pca
    pca = PCA(n_components=50, random_state=42)
    reduced = pca.fit_transform(embeddings)

    print("PCA output shape:", reduced.shape)
    np.save(OUT_PCA_PATH, reduced)

    # --- KMeans clustering ---
    print(f"Clustering with KMeans (k={N_CLUSTERS})...")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(reduced)

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c} words")

    np.save(OUT_CLUSTER_PATH, labels)
    print("Cluster labels saved.")


if __name__ == "__main__":
    main()
