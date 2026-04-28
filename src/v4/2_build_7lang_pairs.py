import pandas as pd
from lingpy import LexStat, Wordlist

#conf
FORMS_PATH = "data/v4_data/forms.csv"
OUTPUT_PATH = "data/v4_data/7lang_cognate_pairs.csv"

LANGUAGES = [
    "rus", "pol", "ukr", "ces",
    "fra", "ita", "spa"
]

THRESHOLD = 0.65

#load
df = pd.read_csv(FORMS_PATH)

df = df[df["Language_ID"].isin(LANGUAGES)]
df = df[df["Value"].notna()]
df = df[df["Form"].notna()]

print("Rows after filtering:", len(df))

# wordlist
data = {}
data[0] = ["doculect", "concept", "value", "ipa"]

for i, row in enumerate(df.itertuples(), start=1):
    data[i] = [
        row.Language_ID,     # language
        row.Parameter_ID,    # concept
        row.Value,           # orthographic form
        row.Form             # IPA transcription
    ]

wl = Wordlist(data)

#lexstat
lex = LexStat(wl)
lex.get_scorer()
lex.cluster(method="lexstat", threshold=THRESHOLD, ref="lexstat")

print("LexStat clustering done.")

#pairwise combinations
pairs = []

concepts = set(lex[idx, "concept"] for idx in lex)

for concept in concepts:

    entries = [
        (
            lex[idx, "doculect"],
            lex[idx, "value"],  # orthographic form
            lex[idx, "ipa"],    # IPA
            lex[idx, "lexstat"]
        )
        for idx in lex
        if lex[idx, "concept"] == concept
    ]

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):

            lang1, form1, ipa1, cluster1 = entries[i]
            lang2, form2, ipa2, cluster2 = entries[j]

            label = 1 if cluster1 == cluster2 else 0

            pairs.append({
                "concept": concept,
                "lang1": lang1,
                "form1": form1,   # orthography
                "ipa1": ipa1,
                "lang2": lang2,
                "form2": form2,   # orthography
                "ipa2": ipa2,
                "label": label
            })

df_pairs = pd.DataFrame(pairs)

print("Total pairs:", len(df_pairs))
print("Positives:", (df_pairs["label"] == 1).sum())
print("Negatives:", (df_pairs["label"] == 0).sum())

df_pairs.to_csv(OUTPUT_PATH, index=False)

print("Saved to:", OUTPUT_PATH)