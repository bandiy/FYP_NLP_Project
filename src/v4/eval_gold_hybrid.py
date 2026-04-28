import numpy as np
import pandas as pd
import torch
import panphon
from panphon.distance import Distance
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as lev

DATA_PATH = "data/v4_data/7lang_features_with_semantics.csv"
PAIRS_PATH = "data/v4_data/7lang_cognate_pairs.csv"

df = pd.read_csv(DATA_PATH)
pairs_df = pd.read_csv(PAIRS_PATH)

X = df[
    [
        "lev_norm",
        "panphon_dist",
        "phon_cosine",
        "phon_euclid",
        "semantic_sim"
    ]
]

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# phonetic model
clf_phon = RandomForestClassifier(n_estimators=300, random_state=42)
clf_phon.fit(X_train[
    ["lev_norm", "panphon_dist", "phon_cosine", "phon_euclid"]
], y_train)

# hybrid model
clf_hybrid = RandomForestClassifier(n_estimators=300, random_state=42)
clf_hybrid.fit(X_train, y_train)

# semantic model
MODEL_NAME = "xlm-roberta-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# phonology tools
dist = Distance()
ft = panphon.FeatureTable()

def embed(text):

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]


def ipa_vector(ipa):

    vecs = ft.word_to_vector_list(ipa)

    if len(vecs) == 0:
        return np.zeros(24)

    numeric_vecs = []

    for v in vecs:

        # convert + - 0 to numbers
        numeric = [
            1 if x == '+' else
            -1 if x == '-' else
            0
            for x in v
        ]

        numeric_vecs.append(numeric)

    return np.mean(np.array(numeric_vecs, dtype=float), axis=0)


gold_pairs = {
    "брат": "frère",
    "мать": "mère",
    "отец": "père",
    "сын": "fils",
    "зуб": "dent",
    "нос": "nez",
    "имя": "nom",
    "ночь": "nuit",
    "солнце": "soleil",
    "земля": "terre",
    "жить": "vivre",
    "умереть": "mourir",
    "видеть": "voir",
    "дать": "donner",
    "новый": "neuf",
    "брать": "prendre",
    "знать": "connaître"
}

print("\n=== GOLD PAIR PROBABILITY COMPARISON ===\n")

for ru_word, fr_word in gold_pairs.items():

    print("Searching:", ru_word, fr_word)

    match = pairs_df[
        ((pairs_df["form1"] == ru_word) & (pairs_df["form2"] == fr_word)) |
        ((pairs_df["form1"] == fr_word) & (pairs_df["form2"] == ru_word))
    ]

    if match.empty:
        continue

    ipa1 = match.iloc[0]["ipa1"]
    ipa2 = match.iloc[0]["ipa2"]

    lev_norm = lev(ipa1, ipa2) / max(len(ipa1), len(ipa2), 1)
    pan_dist = dist.feature_edit_distance(ipa1, ipa2)

    vec1 = ipa_vector(ipa1)
    vec2 = ipa_vector(ipa2)

    phon_cos = cosine_similarity(
        vec1.reshape(1,-1),
        vec2.reshape(1,-1)
    )[0][0]

    phon_euclid = np.linalg.norm(vec1 - vec2)

    emb1 = embed(ru_word)
    emb2 = embed(fr_word)

    cosine = cosine_similarity(
        emb1.reshape(1,-1),
        emb2.reshape(1,-1)
    )[0][0]

    semantic_sim = (cosine + 1) / 2

    X_phon = pd.DataFrame(
        [[lev_norm, pan_dist, phon_cos, phon_euclid]],
        columns=["lev_norm","panphon_dist","phon_cosine","phon_euclid"]
    )

    X_hyb = pd.DataFrame(
        [[lev_norm, pan_dist, phon_cos, phon_euclid, semantic_sim]],
        columns=["lev_norm","panphon_dist","phon_cosine","phon_euclid","semantic_sim"]
    )

    phon_prob = clf_phon.predict_proba(X_phon)[0][1]
    hyb_prob = clf_hybrid.predict_proba(X_hyb)[0][1]

    print(f"{ru_word} → {fr_word}")
    print(f"  Phonetic Prob: {phon_prob:.4f}")
    print(f"  Hybrid Prob:   {hyb_prob:.4f}\n")  