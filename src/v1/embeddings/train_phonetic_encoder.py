#python -m src.embeddings.train_phonetic_encoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.v1.embeddings.phonetic_encoder import PhoneticEncoder
from src.v1.embeddings.phonetic_contrastive_dataset import (
    anchors,
    positives,
    negatives,
)
from src.v1.preprocessing.phonetic_preprocessing import char2idx


# init

device = "cpu"

model = PhoneticEncoder(
    vocab_size=len(char2idx),
    emb_dim=32,
    hidden_dim=64,
).to(device)

criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

anchors = anchors.to(device)
positives = positives.to(device)
negatives = negatives.to(device)


# train

epochs = 10
batch_size = 64

model.train()

for epoch in range(epochs):
    perm = torch.randperm(len(anchors))
    total_loss = 0.0

    for i in range(0, len(anchors), batch_size):
        idx = perm[i : i + batch_size]

        a = F.normalize(model(anchors[idx]), dim=1)
        p = F.normalize(model(positives[idx]), dim=1)
        n = F.normalize(model(negatives[idx]), dim=1)

        loss = criterion(a, p, n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} — loss: {total_loss:.3f}")


# save

torch.save(
    model.state_dict(),
    "data/corpora/russian/phonetic_encoder.pt",
)

print("Saved trained phonetic encoder")
