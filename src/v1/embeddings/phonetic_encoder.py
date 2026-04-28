import torch
import torch.nn as nn


class PhoneticEncoder(nn.Module):

    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size + 1,  # +1 for padding
            embedding_dim=emb_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)              # (batch, seq_len, emb_dim)
        _, (h, _) = self.lstm(emb)            # h: (1, batch, hidden_dim)
        return h[-1]                          # (batch, hidden_dim)
