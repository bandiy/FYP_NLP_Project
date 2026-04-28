import random
import torch
import Levenshtein

from src.v1.preprocessing.phonetic_preprocessing import X, words, pairs


# init

word2ipa = dict(pairs)
word_to_idx = {w: i for i, w in enumerate(words)}


def ipa_distance(w1, w2):
    return Levenshtein.distance(word2ipa[w1], word2ipa[w2])


# triplets

def build_triplets(
    num_triplets=5000,
    pos_max=2,
    neg_min=6,
):

    triplets = []
    vocab = words

    while len(triplets) < num_triplets:
        anchor = random.choice(vocab)

        positives = [
            w for w in vocab
            if w != anchor and ipa_distance(anchor, w) <= pos_max
        ]

        negatives = [
            w for w in vocab
            if ipa_distance(anchor, w) >= neg_min
        ]

        if positives and negatives:
            pos = random.choice(positives)
            neg = random.choice(negatives)
            triplets.append((anchor, pos, neg))

    return triplets


# tensors

triplets = build_triplets()
print("Triplets built:", len(triplets))

anchors = torch.tensor([X[word_to_idx[a]] for a, _, _ in triplets])
positives = torch.tensor([X[word_to_idx[p]] for _, p, _ in triplets])
negatives = torch.tensor([X[word_to_idx[n]] for _, _, n in triplets])
