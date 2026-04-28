from pathlib import Path
from collections import Counter

# load word and ipa pairs

def load_word_ipa(path):
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        word, ipa = line.split("\t")
        ipa = ipa.replace("\n", "").strip()
        pairs.append((word, ipa))
    return pairs


pairs = load_word_ipa("data/corpora/russian/word_ipa.tsv")

print(f"Loaded {len(pairs)} word–IPA pairs")
print("Sample:", pairs[:10])

# build ipa vocab

ipa_chars = sorted(set("".join(ipa for _, ipa in pairs)))

PAD_IDX = 0
char2idx = {c: i + 1 for i, c in enumerate(ipa_chars)}
idx2char = {i: c for c, i in char2idx.items()}

print("\nIPA characters:", ipa_chars)
print("IPA vocab size:", len(char2idx))

# encode ipa strings as fixed length sequences

# inspect IPA lengths - justification
lengths = [len(ipa) for _, ipa in pairs]
print("\nIPA length stats:")
print("  max:", max(lengths))
print("  mean:", sum(lengths) / len(lengths))

MAX_LEN = 24  # above max observed length


def encode_ipa(ipa, max_len=MAX_LEN):
    seq = [char2idx[c] for c in ipa if c in char2idx]
    seq = seq[:max_len]
    return seq + [PAD_IDX] * (max_len - len(seq))


X = [encode_ipa(ipa) for _, ipa in pairs]
words = [word for word, _ in pairs]

# sanity checks
assert len(X) == len(words)
assert all(len(seq) == MAX_LEN for seq in X)

print("\nEncoded matrix shape:")
print(f"  {len(X)} × {len(X[0])}")

print("\nExample encoding:")
print(words[0], "→", pairs[0][1], "→", X[0])

#for w, ipa in pairs:
    #if "\n" in ipa:
        #print("FOUND NEWLINE:", repr(ipa))
        #break
#else:
   #print("No newlines in IPA strings")

