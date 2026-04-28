import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/v4_data/7lang_features_with_semantics.csv"

# load

df = pd.read_csv(DATA_PATH)

X_all = df[
    [
        "lev_norm",
        "panphon_dist",
        "phon_cosine",
        "phon_euclid",
        "semantic_sim"
    ]
]

y = df["label"]

# split

X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# scale

scaler = StandardScaler()

X_train_all = scaler.fit_transform(X_train_all)
X_test_all = scaler.transform(X_test_all)

# splits

# column order:
# 0: lev_norm
# 1: panphon_dist
# 2: phon_cosine
# 3: phon_euclid
# 4: semantic_sim

# phonetic only
X_train_phon = X_train_all[:, :4]
X_test_phon = X_test_all[:, :4]

# semantic only
X_train_sem = X_train_all[:, 4].reshape(-1, 1)
X_test_sem = X_test_all[:, 4].reshape(-1, 1)

# phon

clf_phon = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

clf_phon.fit(X_train_phon, y_train)

print("\n==============================")
print("PHONETIC ONLY")
print("==============================")
print(classification_report(y_test, clf_phon.predict(X_test_phon)))

# sem
clf_sem = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

clf_sem.fit(X_train_sem, y_train)

print("\n==============================")
print("SEMANTIC ONLY")
print("==============================")
print(classification_report(y_test, clf_sem.predict(X_test_sem)))

# hybrid

clf_hybrid = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

clf_hybrid.fit(X_train_all, y_train)

print("\n==============================")
print("HYBRID (PHONETIC + SEMANTIC)")
print("==============================")
print(classification_report(y_test, clf_hybrid.predict(X_test_all)))

#feature

importances = clf_hybrid.feature_importances_

feature_names = [
    "lev_norm",
    "panphon_dist",
    "phon_cosine",
    "phon_euclid",
    "semantic_sim"
]

print("\nFeature Importances (Hybrid Model):")

for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")