import numpy as np
import umap
import matplotlib.pyplot as plt

DIST_PATH = "data/corpora/russian2/phonetic_distance_matrix_6000.npy"

print("Loading distance matrix...")
D = np.load(DIST_PATH)

print("Distance matrix shape:", D.shape)

print("Running UMAP on precomputed distances...")

reducer = umap.UMAP(
    metric="precomputed",
    n_neighbors=25,
    min_dist=0.2,
    random_state=42
)

embedding_2d = reducer.fit_transform(D)

print("UMAP output shape:", embedding_2d.shape)

plt.figure(figsize=(10, 8))
plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    s=8,
    alpha=0.7
)

plt.title("Edit-Distance Phonetic Space (UMAP)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()
