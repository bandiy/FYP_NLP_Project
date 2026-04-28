import pandas as pd
import random

INPUT_PATH = "data/v4_data/7lang_cognate_pairs.csv"
OUTPUT_PATH = "data/v4_data/7lang_cognate_pairs_with_semantic_negatives.csv"

df = pd.read_csv(INPUT_PATH)

# wordlevel table
words = []

for row in df.itertuples():
    words.append((row.concept, row.lang1, row.form1, row.ipa1))
    words.append((row.concept, row.lang2, row.form2, row.ipa2))

df_words = pd.DataFrame(words, columns=["concept", "lang", "form", "ipa"])
df_words = df_words.drop_duplicates()

# concept group
concept_groups = {
    concept: group
    for concept, group in df_words.groupby("concept")
}

concept_list = list(concept_groups.keys())

# negative gen
cross_negatives = []

num_to_generate = len(df) // 2  # ratio

for _ in range(num_to_generate):

    # diff concept
    c1, c2 = random.sample(concept_list, 2)

    w1 = concept_groups[c1].sample(1).iloc[0]
    w2 = concept_groups[c2].sample(1).iloc[0]

    cross_negatives.append({
        "concept": f"{c1}__{c2}",  # combined concept
        "lang1": w1["lang"],
        "form1": w1["form"],
        "ipa1": w1["ipa"],
        "lang2": w2["lang"],
        "form2": w2["form"],
        "ipa2": w2["ipa"],
        "label": 0
    })

df_cross = pd.DataFrame(cross_negatives)

# combine
df_final = pd.concat([df, df_cross], ignore_index=True)

# shuffle
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# 🔹 save
df_final.to_csv(OUTPUT_PATH, index=False)

print("Saved dataset:", OUTPUT_PATH)
print("Original rows:", len(df))
print("New rows:", len(df_final))
print("Added cross-concept negatives:", len(df_cross))
print("\nLabel distribution:")
print(df_final["label"].value_counts())