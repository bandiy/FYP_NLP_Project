from pathlib import Path
import epitran

LEMMA_PATH = Path("data/corpora/russian_french/processed/russian_lemma_frequencies.tsv")
OUT_PATH = Path("data/corpora/russian_french/processed/russian_lemma_ipa.tsv")

epi = epitran.Epitran("rus-Cyrl")

def main():

    with LEMMA_PATH.open(encoding="utf-8") as f:
        lines = f.readlines()

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for line in lines:
            lemma, _ = line.strip().split("\t")
            ipa = epi.transliterate(lemma)
            out.write(f"{lemma}\t{ipa}\n")

    print(f"Saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
