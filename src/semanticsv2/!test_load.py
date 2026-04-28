
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/corpora/russian2/lemma_frequencies.tsv")

def load_lemmas(min_freq=30):
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["lemma", "freq"])
    
    print("Total lemmas:", len(df))
    
    df = df[df["freq"] >= min_freq]
    
    print("After frequency filter:", len(df))
    print("First 10 lemmas:")
    print(df.head(10))
    
    return df["lemma"].tolist()

def main():
    lemmas = load_lemmas()
    print("\nLoaded", len(lemmas), "lemmas successfully.")

if __name__ == "__main__":
    main()
