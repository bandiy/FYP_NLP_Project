import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

import panphon
from panphon.distance import Distance
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as lev


# config
#no neg imput path
#INPUT_PATH = "data/github/7lang_cognate_pairs.csv"

#with neg input path
#INPUT_PATH = "data/github/7lang_cognate_pairs_with_semantic_negatives.csv"

#with hard neg input path
INPUT_PATH = "data/v4_data/7lang_with_hard_negatives.csv"

OUTPUT_PATH = "data/v4_data/7lang_features_with_semantics.csv"
#older
#MODEL_NAME = "xlm-roberta-base"

#newer
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading SentenceTransformer model...")
model = SentenceTransformer(MODEL_NAME, device=device) # type: ignore

# phon
dist = Distance()
ft = panphon.FeatureTable()

# func
def get_embedding(text):
    return model.encode(text)


def ipa_vector(ipa):
    vecs = ft.word_to_vector_list(ipa, numeric=True)

    if len(vecs) == 0:
        return np.zeros(24)

    return np.mean(vecs, axis=0)


# load

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

df = df.dropna(subset=["form1", "form2", "ipa1", "ipa2"])

embedding_cache = {}

def cached_embedding(word):
    if word not in embedding_cache:
        embedding_cache[word] = get_embedding(word)
    return embedding_cache[word]


# extract
features = []

print("Extracting phonetic + semantic features...")

for row in tqdm(df.itertuples(), total=len(df)):

    word1 = str(row.form1)
    word2 = str(row.form2)

    ipa1 = str(row.ipa1)
    ipa2 = str(row.ipa2)

    # phon

    lev_dist = lev(ipa1, ipa2)
    lev_norm = lev_dist / max(len(ipa1), len(ipa2), 1)

    pan_dist = dist.feature_edit_distance(ipa1, ipa2)

    vec1 = ipa_vector(ipa1)
    vec2 = ipa_vector(ipa2)

    phon_cos = cosine_similarity(
        vec1.reshape(1, -1),
        vec2.reshape(1, -1)
    )[0][0]

    phon_euclid = np.linalg.norm(vec1 - vec2)

    # improved sem

    emb1 = cached_embedding(word1)
    emb2 = cached_embedding(word2)

    cosine = cosine_similarity(
        emb1.reshape(1, -1),
        emb2.reshape(1, -1)
    )[0][0]

    # normalize to [0,1]
    semantic_sim = (cosine + 1) / 2

    # store

    features.append({
        "lev_norm": lev_norm,
        "panphon_dist": pan_dist,
        "phon_cosine": phon_cos,
        "phon_euclid": phon_euclid,
        "semantic_sim": semantic_sim,
        "label": row.label
    })


#save

df_feat = pd.DataFrame(features)
df_feat.to_csv(OUTPUT_PATH, index=False)

print("\nSaved features to:", OUTPUT_PATH)
print("Total rows:", len(df_feat))