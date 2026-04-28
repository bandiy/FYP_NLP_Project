import re
from pathlib import Path

# load raw Russian text
text = Path("data/corpora/russian/crime_and_punishment.txt").read_text(encoding="utf-8")

# normalise
text = text.lower()
text = re.sub(r"[^а-яё\s]", " ", text)
text = re.sub(r"\s+", " ", text)

tokens = text.split()

print("Total tokens:", len(tokens))
print("Sample tokens:", tokens[:20])

# save tokens
Path("data/corpora/russian/tokens.txt").write_text(
    "\n".join(tokens),
    encoding="utf-8"
)
