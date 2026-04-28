from pathlib import Path
import epitran

#config

LEMMA_PATH = Path("data/corpora/russian2/lemma_frequencies.tsv")
OUT_PATH = Path("data/corpora/russian2/lemma_ipa.tsv")

TOP_N = 6000  # 3000

#epitran

epi = epitran.Epitran("rus-Cyrl")

#main

def build_ipa_lexicon():
    print("Loading lemmas...")

    lemmas = []

    with LEMMA_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= TOP_N:
                break
            lemma, _ = line.strip().split("\t")
            lemmas.append(lemma)

    print(f"Loaded {len(lemmas)} lemmas")

    print("Converting to IPA...")

    ipa_map = {}

    for lemma in lemmas:
        ipa = epi.transliterate(lemma)
        ipa_map[lemma] = ipa

    print("Saving IPA lexicon...")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for lemma, ipa in ipa_map.items():
            f.write(f"{lemma}\t{ipa}\n")

    print("Done.")
    print(f"Saved to: {OUT_PATH}")

if __name__ == "__main__":
    build_ipa_lexicon()
