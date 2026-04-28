import epitran
from pathlib import Path

epi = epitran.Epitran("rus-Cyrl")

words = Path("data/corpora/russian/vocab.txt").read_text(
    encoding="utf-8"
).splitlines()

pairs = []

for w in words:
    ipa = epi.transliterate(w)
    if ipa.strip():
        pairs.append((w, ipa))

print("Total words:", len(words))
print("IPA converted:", len(pairs))
print("Sample:", pairs[:10])

out = "\n".join(f"{w}\t{ipa}" for w, ipa in pairs)
Path("data/corpora/russian/word_ipa.tsv").write_text(
    out,
    encoding="utf-8"
)
