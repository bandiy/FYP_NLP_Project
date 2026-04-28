#   python -m src.visualisations.phonetic_semantic_umap

import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import normalize

# config
ALPHA = 0.4
SEED = 42
DEVICE = "cpu"

#CRIME AND PUNISHMENT
WORDS_TO_LABEL = ["мысль",
    "страх",
    "человек",
    "комната",
    "вина",]

# diff
#WORDS_TO_LABEL = ["государство", "европе", "северной"]

# same semantics
#WORDS_TO_LABEL = ["государство", "конституция", "президент"]

# similar phonetics
#WORDS_TO_LABEL = ["северной", "южной", "восточной"]

# geographical
#WORDS_TO_LABEL = ["восточной", "европе", "северной"]

# same root
#WORDS_TO_LABEL = ["россия", "российский", "россияне"]

# abstract but government
#WORDS_TO_LABEL = ["федерация", "республика", "государство"]

# demography but different abstraction
#WORDS_TO_LABEL = ["население", "страны", "россияне"]

# intl relations
#WORDS_TO_LABEL = ["международных", "организаций", "оон"]

# diachronic - old vs new
#WORDS_TO_LABEL = ["киевская", "русь", "государство"]

PHON_PATH = "data/corpora/russian/phonetic_embeddings.pt"
SEM_PATH = "data/corpora/russian/semantic_embeddings.pt"

# helpers
def safe_load(path):
    return torch.load(path, map_location=DEVICE, weights_only=False)

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def reduce_umap(vecs):
    return umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=SEED,
    ).fit_transform(vecs)

# load

phonetic = safe_load(PHON_PATH)
semantic = safe_load(SEM_PATH)

words = phonetic["words"]

phon_vecs = to_numpy(phonetic["embeddings"])
sem_vecs = to_numpy(semantic["embeddings"])

# normalise
phon_vecs = normalize(phon_vecs, axis=1)
sem_vecs = normalize(sem_vecs, axis=1)

# semantic 

PHON_DIM = phon_vecs.shape[1]
SEM_DIM = sem_vecs.shape[1]

projector = torch.nn.Linear(SEM_DIM, PHON_DIM, bias=False)
projector.eval()

with torch.no_grad():
    sem_proj = projector(torch.from_numpy(sem_vecs).float()).numpy()

# fusion

fused_vecs = ALPHA * phon_vecs + (1 - ALPHA) * sem_proj
fused_vecs = normalize(fused_vecs, axis=1)

# umap

phon_2d = reduce_umap(phon_vecs)
sem_2d = reduce_umap(sem_vecs)
fused_2d = reduce_umap(fused_vecs)

# plot

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

spaces = [
    ("Phonetic embedding space (UMAP)", phon_2d),
    ("Semantic embedding space (UMAP)", sem_2d),
    (f"Fused embedding space (UMAP, α = {ALPHA})", fused_2d),
]

for ax, (title, data) in zip(axes, spaces):
    ax.scatter(data[:, 0], data[:, 1], s=12, alpha=0.6)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")

    for word in WORDS_TO_LABEL:
        idx = words.index(word)
        ax.scatter(data[idx, 0], data[idx, 1], s=80)
        ax.text(
            data[idx, 0],
            data[idx, 1],
            word,
            fontsize=11,
            weight="bold",
        )

plt.tight_layout()
plt.show()
