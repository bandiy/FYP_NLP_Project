# blue -> green -> yellow -> red - increasing phonetic influence
#   python -m src.visualisations.alpha_sweep_umap

import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import normalize
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# config
SEED = 42
DEVICE = "cpu"

ALPHAS = np.linspace(0.0, 1.0, 11)

# diff
#WORDS_TO_TRACK = ["государство", "европе", "северной"]

# same root
#WORDS_TO_TRACK = ["россия", "российский", "россияне"]

# diachronic - old vs new
#WORDS_TO_TRACK = ["киевская", "русь", "государство"]

#CRIME AND PUNISHMENT
WORDS_TO_TRACK = ["мысль",
    "страх",
    "человек",
    "комната",
    "вина",]

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

# embeddings

phonetic = safe_load(PHON_PATH)
semantic = safe_load(SEM_PATH)

words = phonetic["words"]

phon_vecs = normalize(to_numpy(phonetic["embeddings"]), axis=1)
sem_vecs = normalize(to_numpy(semantic["embeddings"]), axis=1)

# sem and phon

PHON_DIM = phon_vecs.shape[1]
SEM_DIM = sem_vecs.shape[1]

projector = torch.nn.Linear(SEM_DIM, PHON_DIM, bias=False)
projector.eval()

with torch.no_grad():
    sem_proj = projector(torch.from_numpy(sem_vecs).float()).numpy()

# fused

fused_by_alpha = {}

for alpha in ALPHAS:
    fused = alpha * phon_vecs + (1 - alpha) * sem_proj
    fused = normalize(fused, axis=1)
    fused_by_alpha[alpha] = fused

# umap

stacked = np.vstack(list(fused_by_alpha.values()))
umap_2d = reduce_umap(stacked)

n = phon_vecs.shape[0]
umap_by_alpha = {}
i = 0
for alpha in ALPHAS:
    umap_by_alpha[alpha] = umap_2d[i : i + n]
    i += n

# plot

fig, ax = plt.subplots(figsize=(8, 8))

# background: phonetic-dominant space
bg = umap_by_alpha[1.0]
ax.scatter(bg[:, 0], bg[:, 1], s=8, alpha=0.15)

# colour mapping for α
norm = Normalize(vmin=ALPHAS.min(), vmax=ALPHAS.max())
cmap = cm.viridis

# plot trajectories
for word in WORDS_TO_TRACK:
    idx = words.index(word)

    points = np.array([
        umap_by_alpha[alpha][idx] for alpha in ALPHAS
    ])

    segments = np.stack([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=3,
        alpha=0.9,
    )
    lc.set_array(ALPHAS[:-1])
    ax.add_collection(lc)

    # arrows to indicate direction
    for i in range(len(points) - 1):
        ax.annotate(
            "",
            xy=points[i + 1],
            xytext=points[i],
            arrowprops=dict(
                arrowstyle="->",
                color=cmap(norm(ALPHAS[i])),
                lw=1.5,
                alpha=0.9,
            ),
        )

    # label at phonetic end (α = 1)
    ax.text(
        points[-1, 0],
        points[-1, 1],
        word,
        fontsize=11,
        weight="bold",
    )

# colourbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("α (semantic → phonetic)", fontsize=11)

ax.set_title(
    "α-sweep trajectories in fused embedding space (UMAP)",
    fontsize=14,
)
ax.set_xlabel("UMAP dimension 1")
ax.set_ylabel("UMAP dimension 2")

plt.tight_layout()
plt.show()
