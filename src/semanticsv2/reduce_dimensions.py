import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path

PCA_PATH = Path("data/corpora/russian2/semantic_pca.npy")
OUT_UMAP_PATH = Path("data/corpora/russian2/semantic_umap.npy")
OUT_IMG_PATH = Path("data/corpora/russian2/semantic_umap.png")

def main():
    # load PCA-reduced data
    reduced = np.load(PCA_PATH)
    print("Shape:", reduced.shape)

    # create UMAP reducer
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )

    # fit and transform
    embedding_2d = reducer.fit_transform(reduced)
    print("UMAP output shape:", embedding_2d.shape)

    # save raw UMAP data
    np.save(OUT_UMAP_PATH, embedding_2d)
    print("UMAP projection saved as .npy")

    # Plot and save as image
    plt.figure(figsize=(8, 6))
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        s=5,
        alpha=0.7
    )

    plt.title("UMAP Projection")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    plt.tight_layout()
    plt.savefig(OUT_IMG_PATH, dpi=300)
    plt.show()
    plt.close()

    print(f"UMAP plot saved to {OUT_IMG_PATH}")

if __name__ == "__main__":
    main()