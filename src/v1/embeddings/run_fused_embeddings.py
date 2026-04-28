#   python -m src.embeddings.run_fused_embeddings

import torch
from torch.nn.functional import cosine_similarity, normalize

# load phonetic embeddings
phonetic = torch.load(
    "data/corpora/russian/phonetic_embeddings.pt",
    map_location="cpu",
    weights_only=False,
)

semantic = torch.load(
    "data/corpora/russian/semantic_embeddings.pt",
    map_location="cpu",
    weights_only=False,
)

words = phonetic["words"]

phonetic_vecs = phonetic["embeddings"]
semantic_vecs = semantic["embeddings"]

# ensure tensors
if not torch.is_tensor(phonetic_vecs):
    phonetic_vecs = torch.tensor(phonetic_vecs)

if not torch.is_tensor(semantic_vecs):
    semantic_vecs = torch.tensor(semantic_vecs)

# normalize each modality
phonetic_vecs = normalize(phonetic_vecs, dim=1)
semantic_vecs = normalize(semantic_vecs, dim=1)

# fuse by concatenation
alpha = 0.4  # phonetic weight (0.3, 0.5, 0.7)
# 0.7 keeps high morphological words - high phonological
# 0.5 balanced
# 0.3 is very conceptual - high semantics


fused_vecs = torch.cat(
    [alpha * phonetic_vecs, (1 - alpha) * semantic_vecs],
    dim=1,
)

print("Fused embedding shape:", fused_vecs.shape)
# expected: (1276, 448)


def nearest(word, k=5):
    idx = words.index(word)
    sims = cosine_similarity(
        fused_vecs[idx].unsqueeze(0),
        fused_vecs,
    )
    top = sims.argsort(descending=True)[1 : k + 1]
    return [(words[i], round(sims[i].item(), 3)) for i in top]


print("\nNearest neighbours (FUSED):")
for test_word in ["государство", "европе", "северной"]:
    print(test_word, "→", nearest(test_word))

#print("\nNearest neighbours (fused):")
#for test_word in [
#    "мысль",
#    "страх",
#    "человек",
#    "комната",
#    "вина",
#]:
#    print(test_word, "→", nearest(test_word))


# save fused embeddings
torch.save(
    {
        "words": words,
        "embeddings": fused_vecs,
    },
    "data/corpora/russian/fused_embeddings.pt",
)

print("\nSaved fused embeddings.")
