import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

# config

BOOKS = {
    "crime_and_punishment": {
        "base_url": "https://ilibrary.ru/text/69/p.{}/index.html",
        "chapters": 41
    },
    "the_idiot": {
        "base_url": "https://ilibrary.ru/text/94/p.{}/index.html",
        "chapters": 50
    },
    "humiliated_and_insulted": {
        "base_url": "https://ilibrary.ru/text/64/p.{}/index.html",
        "chapters": 46
    },
    "the_brothers_karamazov": {
        "base_url": "https://ilibrary.ru/text/1199/p.{}/index.html",
        "chapters": 96
    }
}

OUT_DIR = Path("data/corpora/russian2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# scraping
def fetch_book(book_name, config):
    print(f"\n=== Fetching: {book_name} ===")

    all_text = []
    base_url = config["base_url"]
    num_chapters = config["chapters"]

    for chapter in range(1, num_chapters + 1):
        url = base_url.format(chapter)
        print(f"Fetching chapter {chapter}/{num_chapters}")

        r = requests.get(url, timeout=20)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        text_div = soup.find("div", id="text")
        if text_div is None:
            raise RuntimeError(f"No text div found in {book_name}, chapter {chapter}")

        paragraphs = text_div.find_all("z")
        if not paragraphs:
            raise RuntimeError(f"No <z> tags found in {book_name}, chapter {chapter}")

        chapter_text = "\n".join(
            z.get_text(strip=True) for z in paragraphs
        )

        all_text.append(chapter_text)

        time.sleep(0.5)

    out_path = OUT_DIR / f"{book_name}.txt"
    out_path.write_text("\n\n".join(all_text), encoding="utf-8")

    print(f"Saved to: {out_path}")
    print(f"Total characters: {len(''.join(all_text))}")


#main

if __name__ == "__main__":
    for book_name, config in BOOKS.items():
        fetch_book(book_name, config)

    print("\nAll books downloaded successfully.")
