import pandas as pd
from lingpy import LexStat, Wordlist

# -----------------------------
# CONFIG
# -----------------------------
FORMS_PATH = "data/v4_data/forms.csv"
LANGUAGES = [
    "rus", "pol", "ukr", "ces",
    "fra", "ita", "spa"
]

THRESHOLDS = [0.55, 0.6, 0.65, 0.7]

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(FORMS_PATH)

df = df[df["Language_ID"].isin(LANGUAGES)]
df = df[df["Form"].notna()]

print("Rows after filtering:", len(df))

# -----------------------------
# BUILD LINGPY DATA STRUCTURE
# -----------------------------
data = {}
data[0] = ["doculect", "concept", "ipa"]

for i, row in enumerate(df.itertuples(), start=1):
    data[i] = [
        row.Language_ID,
        row.Parameter_ID,
        row.Form
    ]

wl = Wordlist(data)

# -----------------------------
# RUN FOR MULTIPLE THRESHOLDS
# -----------------------------
for threshold in THRESHOLDS:

    print("\n==============================")
    print(f"Threshold: {threshold}")
    print("==============================")

    lex = LexStat(wl)
    lex.get_scorer()
    lex.cluster(method="lexstat", threshold=threshold, ref="lexstat")

    # Extract Russian–French pairs
    pairs = []

    for concept in set(lex[idx, "concept"] for idx in lex):
        entries = [
            (lex[idx, "doculect"], lex[idx, "ipa"], lex[idx, "lexstat"])
            for idx in lex
            if lex[idx, "concept"] == concept
        ]

        ru = [e for e in entries if e[0] == "rus"]
        fr = [e for e in entries if e[0] == "fra"]

        if not ru or not fr:
            continue

        for r in ru:
            for f in fr:
                label = 1 if r[2] == f[2] else 0
                pairs.append(label)

    total = len(pairs)
    positives = sum(pairs)
    negatives = total - positives

    print(f"Total RF pairs: {total}")
    print(f"Positives: {positives}")
    print(f"Negatives: {negatives}")
    if total > 0:
        print(f"Positive rate: {positives / total:.4f}")