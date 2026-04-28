import re
from pathlib import Path

# load text

text = Path("data/corpora/russian/raw.txt").read_text(
    encoding="utf-8"
)

# remove line breaks

text = re.sub(r"\n+", " ", text)
text = re.sub(r"={2,}.*?={2,}", " ", text)
text = re.sub(r"\s+", " ", text)


# abbreviations

ABBREVIATIONS = [
    "сокр.",
    "г.",
    "км.",
    "км2.",
    "млн.",
    "млрд.",
]

for abbr in ABBREVIATIONS:
    safe = abbr.replace(".", "<DOT>")
    text = text.replace(abbr, safe)

# split

sentences = re.split(
    r"(?<=[.!?])\s+(?=[А-ЯЁ0-9])",
    text
)

# abbreviations

sentences = [
    s.replace("<DOT>", ".").strip().lower()
    for s in sentences
    if len(s.strip()) > 20
]

# output
print("Total sentences:", len(sentences))
print("Sample sentences:")
for s in sentences[:5]:
    print("-", s)

Path("data/corpora/russian/sentences.txt").write_text(
    "\n".join(sentences),
    encoding="utf-8"
)
