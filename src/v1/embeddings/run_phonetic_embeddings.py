#   python -m src.embeddings.run_phonetic_embeddings

import torch
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

from src.v1.preprocessing.phonetic_preprocessing import X, words, char2idx
from src.v1.embeddings.phonetic_encoder import PhoneticEncoder

USE_TRAINED_MODEL = True   # if alreayd trained

import os

if USE_TRAINED_MODEL:
    path = "data/corpora/russian/phonetic_encoder.pt"
    if not os.path.exists(path):
        raise RuntimeError(
            "Trained phonetic encoder not found. "
            "Run: python -m src.embeddings.train_phonetic_encoder"
        )

# setup

device = "cpu"

X_tensor = torch.tensor(X, dtype=torch.long).to(device)

model = PhoneticEncoder(
    vocab_size=len(char2idx),
    emb_dim=32,
    hidden_dim=64,
).to(device)

if USE_TRAINED_MODEL:
    model.load_state_dict(
        torch.load(
            "data/corpora/russian/phonetic_encoder.pt",
            map_location=device,
        )
    )
    print("Using TRAINED phonetic encoder")
else:
    print("Using UNTRAINED phonetic encoder (baseline)")

model.eval()


# generate phonetic embeddings

with torch.no_grad():
    phonetic_vectors = F.normalize(model(X_tensor), dim=1)

print("Phonetic embedding shape:", phonetic_vectors.shape)
# expected: (1276, 64)


# nearest neighbours

def nearest(word, k=5):
    idx = words.index(word)
    sims = cosine_similarity(
        phonetic_vectors[idx].unsqueeze(0),
        phonetic_vectors,
    )
    top = sims.argsort(descending=True)[1 : k + 1]
    return [(words[i], sims[i].item()) for i in top]



print(
    "Max pairwise diff:",
    torch.max(
        torch.norm(
            phonetic_vectors[1:] - phonetic_vectors[:-1],
            dim=1
        )
    ).item()
)

#print("\nNearest neighbours (phonetic):")
#for test_word in [
    #"человек",
    #"мысль",
    #"страх",
    #"комната",
    #"пошел",
#]:
    #print(test_word, "→", nearest(test_word))

print("\nNearest neighbours (phonetic):") 
for test_word in ["государство", "европе", "северной"]: 
    print(test_word, "→", nearest(test_word))


# save embeddings

torch.save(
    {
        "words": words,
        "embeddings": phonetic_vectors,
        "trained": USE_TRAINED_MODEL,
    },
    "data/corpora/russian/phonetic_embeddings.pt",
)

print("\nSaved phonetic embeddings.")
