# python -m src.embeddings.sentences.run_sentence_semantic_embeddings
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

# load

sentences = Path(
    "data/corpora/russian/sentences.txt"
).read_text(encoding="utf-8").splitlines()

print("Loaded sentences:", len(sentences))

# load sentence model

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# encode sentences

embeddings = model.encode(
    sentences,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

# save

torch.save(
    {
        "sentences": sentences,
        "embeddings": embeddings,
    },
    "data/corpora/russian/sentence_semantic_embeddings.pt",
)

print("Saved sentence semantic embeddings:", embeddings.shape)
