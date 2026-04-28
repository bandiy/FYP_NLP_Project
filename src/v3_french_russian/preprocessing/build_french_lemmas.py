from pathlib import Path
from collections import Counter
import re
import spacy

# config
CORPUS_DIR = Path("data/corpora/russian_french/corpora/french")
OUT_PATH = Path("data/corpora/russian_french/processed/french_lemma_frequencies.tsv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TOP_N = 7000
FINAL_N = 5000
MIN_LENGTH = 3
FILTER_POS = True

nlp = spacy.load("fr_core_news_md")

# use spaCy built-in stopwords
french_stopwords = nlp.Defaults.stop_words


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)          # remove numbers
    text = re.sub(r"[^\w\sàâçéèêëîïôûùüÿñæœ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_lemma_list():

    all_text = ""

    for file_path in CORPUS_DIR.glob("*.txt"):
        print(f"Reading {file_path.name}")
        all_text += file_path.read_text(encoding="utf-8") + " "

    cleaned = clean_text(all_text)


    lemmas = []

    CHUNK_SIZE = 500_000  # 500k characters

    for i in range(0, len(cleaned), CHUNK_SIZE):
        chunk = cleaned[i:i+CHUNK_SIZE]
        doc = nlp(chunk)

        for token in doc:
            if not token.lemma_:
                continue

            lemma = token.lemma_.lower()

            if lemma in french_stopwords:
                continue

            if len(lemma) < MIN_LENGTH:
                continue

            if FILTER_POS and token.pos_ not in {"NOUN", "VERB", "ADJ"}:
                continue

            lemmas.append(lemma)

    print(f"Total candidate lemmas: {len(lemmas)}")

    freq = Counter(lemmas)
    most_common = freq.most_common(TOP_N)

    most_common = most_common[:FINAL_N]

    print(f"Saving top {len(most_common)} lemmas...")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for lemma, count in most_common:
            f.write(f"{lemma}\t{count}\n")

    print(f"Saved to: {OUT_PATH}")



if __name__ == "__main__":
    build_lemma_list()
