import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

BASE_URL = "https://ilibrary.ru/text/69/p.{}/index.html"
NUM_CHAPTERS = 41

OUT_PATH = Path("data/corpora/russian/crime_and_punishment.txt") #Преступление и наказание
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

all_text = []

for chapter in range(1, NUM_CHAPTERS + 1):
    url = BASE_URL.format(chapter)
    print(f"Fetching chapter {chapter}/{NUM_CHAPTERS}")

    r = requests.get(url, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    text_div = soup.find("div", id="text")
    if text_div is None:
        raise RuntimeError(f"No text div found on chapter {chapter}")

    paragraphs = text_div.find_all("z")
    if not paragraphs:
        raise RuntimeError(f"No <z> tags found on chapter {chapter}")

    chapter_text = "\n".join(
        z.get_text(strip=True) for z in paragraphs
    )

    all_text.append(chapter_text)

    time.sleep(0.5)

# Join chapters with a single blank line between them
OUT_PATH.write_text("\n\n".join(all_text), encoding="utf-8")

print("Done.")
print("Saved to:", OUT_PATH)
print("Total characters:", len("".join(all_text)))
