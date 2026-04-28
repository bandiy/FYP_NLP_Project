# python -m src.visualisations.phonetic_semantic_spaces

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

# config

ALPHA = 0.4
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

#CRIME AND PUNISHMENT
WORDS_TO_LABEL = ["мысль",
    "страх",
    "человек",
    "комната",
    "вина",]

DEVICE = "cpu"
SEED = 42

PHON_PATH = "data/corpora/russian/phonetic_embeddings.pt"
SEM_PATH = "data/corpora/russian/semantic_embeddings.pt"

# load 2.6+

def safe_load(path):
    return torch.load(path, map_location=DEVICE, weights_only=False)

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().float()
    else:
        return torch.from_numpy(x).float()

# load embeddings

phonetic = safe_load(PHON_PATH)
semantic = safe_load(SEM_PATH)

words = phonetic["words"]

phon_vecs = to_tensor(phonetic["embeddings"])
sem_vecs = to_tensor(semantic["embeddings"])

# normalise

phon_vecs = torch.tensor(normalize(phon_vecs.numpy(), axis=1))
sem_vecs = torch.tensor(normalize(sem_vecs.numpy(), axis=1))

# phonetic dims

PHON_DIM = phon_vecs.shape[1]
SEM_DIM = sem_vecs.shape[1]

projector = nn.Linear(SEM_DIM, PHON_DIM, bias=False)
projector.eval()

with torch.no_grad():
    sem_proj = projector(sem_vecs)

# fusion

fused_vecs = ALPHA * phon_vecs + (1 - ALPHA) * sem_proj
fused_vecs = torch.tensor(normalize(fused_vecs.numpy(), axis=1))

# t-SNE

def reduce(vecs):
    return TSNE(
        n_components=2,
        perplexity=30,
        random_state=SEED,
        init="pca",
    ).fit_transform(vecs)

phon_2d = reduce(phon_vecs)
sem_2d = reduce(sem_vecs)
fused_2d = reduce(fused_vecs)


# NN list

def nearest(vecs, word, k=5):
    idx = words.index(word)
    sims = np.dot(vecs, vecs[idx])
    top = np.argsort(-sims)[1 : k + 1]
    return [(words[i], float(sims[i])) for i in top]

def print_nn_block(word):
    print("\n" + "=" * 60)
    print(f"WORD: {word}")
    print("-" * 60)

    print("\nPhonetic neighbours:")
    for w, s in nearest(phon_vecs.numpy(), word):
        print(f"  • {w} ({s:.3f})")

    print("\nSemantic neighbours:")
    for w, s in nearest(sem_vecs.numpy(), word):
        print(f"  • {w} ({s:.3f})")

    print(f"\nFused neighbours (α = {ALPHA}):")
    for w, s in nearest(fused_vecs.numpy(), word):
        print(f"  • {w} ({s:.3f})")

    print("=" * 60)

if __name__ == "__main__":
    for word in WORDS_TO_LABEL:
        print_nn_block(word)

# PLOT

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

spaces = [
    ("Phonetic embedding space", phon_2d),
    ("Semantic embedding space", sem_2d),
    (f"Fused embedding space (α = {ALPHA})", fused_2d),
]

for ax, (title, data) in zip(axes, spaces):
    ax.scatter(data[:, 0], data[:, 1], s=12, alpha=0.6)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")

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