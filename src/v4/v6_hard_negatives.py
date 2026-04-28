import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# CONFIG
# ========================

INPUT_PATH = "data/v4_data/7lang_cognate_pairs_with_semantic_negatives.csv"
OUTPUT_PATH = "data/v4_data/7lang_with_hard_negatives.csv"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 10
NEG_PER_ROW = 0.3

# load+clean

print("Loading data...")
df = pd.read_csv(INPUT_PATH)

# remove bad rows
df = df.dropna(subset=["form1", "form2"])

print(f"Rows after cleaning: {len(df)}")

# load model
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

# build vocab

words = list(set(
    df["form1"].dropna().tolist() +
    df["form2"].dropna().tolist()
))

print(f"Total unique words: {len(words)}")

# embeddings

print("Encoding words...")
embeddings = model.encode(words, show_progress_bar=True)
embeddings = np.array(embeddings)

word_to_emb = dict(zip(words, embeddings))
word_index = {w: i for i, w in enumerate(words)}

# sim matrix

print("Computing similarity matrix...")
sim_matrix = cosine_similarity(embeddings)

# =pos pairs

positive_pairs = set()

for row in df.itertuples():
    if row.label == 1:
        positive_pairs.add((str(row.form1), str(row.form2)))
        positive_pairs.add((str(row.form2), str(row.form1)))

# gen hard negs

new_rows = []

print("Generating hard negatives...")

for row in tqdm(df.itertuples(), total=len(df)):

    w1 = str(row.form1)
    concept = row.concept

    if w1 not in word_index:
        continue

    # only apply hard negatives
    if np.random.rand() > NEG_PER_ROW:
        continue

    idx = word_index[w1]
    sims = sim_matrix[idx]

    top_indices = np.argsort(-sims)[1:TOP_K+1]

    for j in top_indices:
        candidate = words[j]

        if candidate == w1:
            continue

        if (w1, candidate) in positive_pairs:
            continue

        new_rows.append({
            "concept": f"{concept}_hardneg",
            "lang1": row.lang1,
            "form1": w1,
            "ipa1": row.ipa1,
            "lang2": row.lang2,
            "form2": candidate,
            "ipa2": row.ipa2,
            "label": 0
        })

        break  # only one per selected row

# mger and save

df_new = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

df_new.to_csv(OUTPUT_PATH, index=False)

print("\nSaved dataset with HARD negatives:", OUTPUT_PATH)
print("Original rows:", len(df))
print("New rows:", len(df_new))
print("Added hard negatives:", len(new_rows))