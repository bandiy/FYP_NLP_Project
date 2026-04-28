import pandas as pd
import numpy as np
import panphon
from panphon.distance import Distance
from Levenshtein import distance as lev
from sklearn.metrics.pairwise import cosine_similarity

# initial no negatives 
# INPUT_PATH = "data/github/7lang_cognate_pairs.csv"

#negatives added:
INPUT_PATH = "data/v4_data/7lang_cognate_pairs_with_semantic_negatives.csv"
OUTPUT_PATH = "data/v4_data/7lang_features_phonvec.csv"

df = pd.read_csv(INPUT_PATH)

dist = Distance()
ft = panphon.FeatureTable()

features = []

def ipa_to_vector(ipa):

    vecs = ft.word_to_vector_list(ipa, numeric=True)

    if len(vecs) == 0:
        return np.zeros(24)

    return np.mean(vecs, axis=0)


for row in df.itertuples():

    ipa1 = str(row.ipa1)
    ipa2 = str(row.ipa2)

    # existing features
    lev_dist = lev(ipa1, ipa2)
    lev_norm = lev_dist / max(len(ipa1), len(ipa2), 1)

    pan_dist = dist.feature_edit_distance(ipa1, ipa2)

    # new phonetic feature vectors
    vec1 = ipa_to_vector(ipa1)
    vec2 = ipa_to_vector(ipa2)

    cos = cosine_similarity(
        vec1.reshape(1, -1),
        vec2.reshape(1, -1)
    )[0][0]

    euclid = np.linalg.norm(vec1 - vec2)

    features.append({
        "lev_norm": lev_norm,
        "panphon_dist": pan_dist,
        "phon_cosine": cos,
        "phon_euclid": euclid,
        "label": row.label
    })


df_feat = pd.DataFrame(features)
df_feat.to_csv(OUTPUT_PATH, index=False)

print("Saved phonetic feature dataset:", OUTPUT_PATH)
print("Rows:", len(df_feat))