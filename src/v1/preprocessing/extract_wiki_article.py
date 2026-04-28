import json
from pathlib import Path

data = json.loads(
    Path("data/corpora/russian/raw.json").read_text(encoding="utf-8")
)

# mediawiki stores pages under dynamic IDs
pages = data["query"]["pages"]
page = next(iter(pages.values()))

text = page["extract"]

Path("data/corpora/russian/raw.txt").write_text(
    text,
    encoding="utf-8"
)

print("Extracted characters:", len(text))
print(text[:500])
