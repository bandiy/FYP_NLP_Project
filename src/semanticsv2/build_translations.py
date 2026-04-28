import json
from pathlib import Path
from transformers import pipeline

LEMMA_PATH = Path("data/corpora/russian2/semantic_lemmas.txt")
OUT_PATH = Path("data/corpora/russian2/lemma_translations1.json")

def main():
    with open(LEMMA_PATH, encoding="utf-8") as f:
        lemmas = [line.strip() for line in f]

    translator = pipeline(
        "translation_ru_to_en",
        model="Helsinki-NLP/opus-mt-ru-en"
    )

    translations = {}

    batch_size = 64
    for i in range(0, len(lemmas), batch_size):
        batch = lemmas[i:i+batch_size]
        results = translator(batch)

        for lemma, result in zip(batch, results):
            translations[lemma] = result["translation_text"]

        print(f"{min(i+batch_size, len(lemmas))}/{len(lemmas)}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)

    print("Translations saved.")

if __name__ == "__main__":
    main()
