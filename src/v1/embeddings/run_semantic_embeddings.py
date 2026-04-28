#   python -m src.embeddings.run_semantic_embeddings

import torch
from torch.nn.functional import cosine_similarity

from src.v1.preprocessing.phonetic_preprocessing import words
from src.v1.embeddings.semantic_encoder import SemanticEncoder


encoder = SemanticEncoder()

print("Encoding semantic embeddings...")
semantic_vectors = encoder.encode(words)

print("Semantic embedding shape:", semantic_vectors.shape)
# expected (vocab_size, 384)


def nearest(word, k=5):
    idx = words.index(word)
    sims = cosine_similarity(
        torch.tensor(semantic_vectors[idx]).unsqueeze(0),
        torch.tensor(semantic_vectors),
    )
    top = sims.argsort(descending=True)[1 : k + 1]
    return [(words[i], round(sims[i].item(), 3)) for i in top]


#print("\nNearest neighbours (semantic):")
#for test_word in [
#    "мысль",
#    "страх",
#    "человек",
#    "комната",
#    "вина",
#]:
#    print(test_word, "→", nearest(test_word))

print("\nNearest neighbours (semantic):") 
for test_word in ["государство", "европе", "северной"]: 
    print(test_word, "→", nearest(test_word))


torch.save(
    {
        "words": words,
        "embeddings": semantic_vectors,
    },
    "data/corpora/russian/semantic_embeddings.pt",
)

print("\nSaved semantic embeddings.")
