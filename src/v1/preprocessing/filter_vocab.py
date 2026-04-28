from collections import Counter
from pathlib import Path

tokens = Path("data/corpora/russian/tokens.txt").read_text(
    encoding="utf-8"
).splitlines()

counts = Counter(tokens)

# keep moderately frequent words
vocab = [w for w, c in counts.items() if c >= 3]

print("Total tokens:", len(tokens))
print("Unique tokens:", len(counts))
print("Filtered vocab:", len(vocab))

Path("data/corpora/russian/vocab.txt").write_text(
    "\n".join(vocab),
    encoding="utf-8"
)
