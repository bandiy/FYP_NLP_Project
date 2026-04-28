import requests
from pathlib import Path
import re

# config
BOOKS = {
    "les_miserables": [
        "https://www.gutenberg.org/cache/epub/17489/pg17489.txt",
        "https://www.gutenberg.org/cache/epub/17493/pg17493.txt",
        "https://www.gutenberg.org/cache/epub/17494/pg17494.txt",
        "https://www.gutenberg.org/cache/epub/17518/pg17518.txt",
        "https://www.gutenberg.org/cache/epub/17519/pg17519.txt",
    ],
    "notre_dame_de_paris": [
        "https://www.gutenberg.org/cache/epub/19657/pg19657.txt"
    ]
}

OUT_DIR = Path("data/corpora/russian_french/corpora/french")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_gutenberg(text):
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start = text.find(start_marker)
    if start != -1:
        text = text[start + len(start_marker):]

    end = text.find(end_marker)
    if end != -1:
        text = text[:end]

    text = re.sub(r"Produced by.*?\n", "", text)

    text = re.sub(r"\*{3,}", "", text)
    
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()



def download_book(book_name, urls):
    print(f"\n=== Downloading: {book_name} ===")

    full_text = ""

    for i, url in enumerate(urls):
        print(f"Fetching part {i+1}/{len(urls)}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()

        cleaned = clean_gutenberg(r.text)
        full_text += cleaned + "\n\n"

    out_path = OUT_DIR / f"{book_name}.txt"
    out_path.write_text(full_text, encoding="utf-8")

    print(f"Saved to: {out_path}")
    print(f"Total characters (cleaned): {len(full_text)}")


def main():
    for book_name, urls in BOOKS.items():
        download_book(book_name, urls)

    print("\nAll French books downloaded successfully.")

if __name__ == "__main__":
    main()
