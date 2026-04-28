# python -m src.visualisations.sentence_umap

import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

#config

SEED = 42

SENTENCE_INDICES = [12, 47, 103, 234, 402]

PHON_PATH = "data/corpora/russian/sentence_phonetic_embeddings.pt"
SEM_PATH = "data/corpora/russian/sentence_semantic_embeddings.pt"
FUSED_PATH = "data/corpora/russian/sentence_fused_embeddings.pt"

#load embeddings

phon = torch.load(PHON_PATH, map_location="cpu", weights_only=False)
sem = torch.load(SEM_PATH, map_location="cpu", weights_only=False)
fused = torch.load(FUSED_PATH, map_location="cpu", weights_only=False)

sentences = fused["sentences"]

def as_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return x

phon_vecs = as_numpy(phon["embeddings"])
sem_vecs = as_numpy(sem["embeddings"])
fused_vecs = as_numpy(fused["embeddings"])


print("Phonetic vectors:", phon_vecs.shape)
print("Semantic vectors:", sem_vecs.shape)
print("Fused vectors:", fused_vecs.shape)
print("Sentences:", len(sentences))


#print

print("\nSelected sentences:\n")
for i, idx in enumerate(SENTENCE_INDICES, start=1):
    print(f"{i}. {sentences[idx]}")

#umap

def reduce(vecs):
    return umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=SEED,
    ).fit_transform(vecs)

phon_2d = reduce(phon_vecs)
sem_2d = reduce(sem_vecs)
fused_2d = reduce(fused_vecs)

#plot

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

spaces = [
    ("Phonetic sentence space (UMAP)", phon_2d),
    ("Semantic sentence space (UMAP)", sem_2d),
    ("Fused sentence space (UMAP)", fused_2d),
]

for ax, (title, data) in zip(axes, spaces):
    ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.35)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")

    for i, idx in enumerate(SENTENCE_INDICES, start=1):
        x, y = data[idx]
        ax.scatter(x, y, s=90)
        ax.text(
            x,
            y,
            str(i),
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
        )

plt.tight_layout()
plt.show()
