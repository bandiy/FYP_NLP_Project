from pathlib import Path
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords

from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    MorphVocab,
    Doc
)


CORPUS_DIR = Path("data/corpora/russian_french/corpora/russian")
OUT_PATH = Path("data/corpora/russian_french/processed/russian_lemma_frequencies.tsv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TOP_N = 7000
FINAL_N = 5000
MIN_LENGTH = 3
FILTER_POS = True



segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

nltk.download("stopwords")
russian_stopwords = set(stopwords.words("russian"))

def clean_text(text):
    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^а-я\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_lemma_list():

    all_text = ""

    for file_path in CORPUS_DIR.glob("*.txt"):
        print(f"Reading {file_path.name}")
        all_text += file_path.read_text(encoding="utf-8") + " "

    cleaned = clean_text(all_text)

    doc = Doc(cleaned)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    lemmas = []

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

        if not token.lemma:
            continue

        lemma = token.lemma

        if lemma in russian_stopwords:
            continue

        if len(lemma) < MIN_LENGTH:
            continue

        if FILTER_POS and token.pos not in {"NOUN", "VERB", "ADJ"}:
            continue

        lemmas.append(lemma)

    print(f"Total candidate lemmas: {len(lemmas)}")

    freq = Counter(lemmas)

    print(f"Selecting top {TOP_N} by frequency...")
    most_common = freq.most_common(TOP_N)

    most_common = most_common[:FINAL_N]

    print(f"Saving top {len(most_common)} lemmas...")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for lemma, count in most_common:
            f.write(f"{lemma}\t{count}\n")

    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    build_lemma_list()
