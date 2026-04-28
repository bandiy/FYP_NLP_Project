#python -m src.embeddings.sentences.run_sentence_phonetic_embeddings


import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize

# load

phonetic = torch.load(
    "data/corpora/russian/phonetic_embeddings.pt",
    map_location="cpu",
)

words = phonetic["words"]
word_vecs = phonetic["embeddings"]

word_to_vec = {
    w: v for w, v in zip(words, word_vecs)
}

# load

sentences = Path(
    "data/corpora/russian/sentences.txt"
).read_text(encoding="utf-8").splitlines()

sentence_vecs = []
kept_sentences = []

# build vectors
for sent in sentences:
    tokens = sent.split()

    vecs = [
        word_to_vec[t]
        for t in tokens
        if t in word_to_vec
    ]

    # skip sentences with too little phonetic coverage
    if len(vecs) < 3:
        continue

    sent_vec = np.mean(vecs, axis=0)
    sentence_vecs.append(sent_vec)
    kept_sentences.append(sent)

sentence_vecs = normalize(
    np.array(sentence_vecs),
    axis=1,
)

# save

torch.save(
    {
        "sentences": kept_sentences,
        "embeddings": sentence_vecs,
    },
    "data/corpora/russian/sentence_phonetic_embeddings.pt",
)

print("Saved sentence phonetic embeddings:", len(kept_sentences))
