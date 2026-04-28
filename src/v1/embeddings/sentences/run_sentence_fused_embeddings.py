import torch
import numpy as np
from sklearn.preprocessing import normalize
import torch.nn as nn

ALPHA = 0.4

# phonetics

phon = torch.load(
    "data/corpora/russian/sentence_phonetic_embeddings.pt",
    map_location="cpu",
    weights_only=False,
)

# -semantics

sem = torch.load(
    "data/corpora/russian/sentence_semantic_embeddings.pt",
    map_location="cpu",
    weights_only=False,
)

# align

phon_map = dict(zip(phon["sentences"], phon["embeddings"]))
sem_map = dict(zip(sem["sentences"], sem["embeddings"]))

common_sentences = [s for s in phon_map if s in sem_map]
print("Common sentences:", len(common_sentences))

phon_vecs = np.array([phon_map[s] for s in common_sentences])
sem_vecs = np.array([sem_map[s] for s in common_sentences])

# normalise

phon_vecs = normalize(phon_vecs, axis=1)
sem_vecs = normalize(sem_vecs, axis=1)

# project

PHON_DIM = phon_vecs.shape[1]   # 64
SEM_DIM = sem_vecs.shape[1]     # 384

projector = nn.Linear(PHON_DIM, SEM_DIM, bias=False)
projector.eval()

with torch.no_grad():
    phon_proj = projector(torch.tensor(phon_vecs, dtype=torch.float32)).numpy()

phon_proj = normalize(phon_proj, axis=1)

# fuse

fused_vecs = normalize(
    ALPHA * phon_proj + (1 - ALPHA) * sem_vecs,
    axis=1,
)

# save

torch.save(
    {
        "sentences": common_sentences,
        "embeddings": fused_vecs,
        "alpha": ALPHA,
    },
    "data/corpora/russian/sentence_fused_embeddings.pt",
)

print("Saved fused sentence embeddings:", fused_vecs.shape)
