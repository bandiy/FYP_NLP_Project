import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = "data/v4_data/7lang_features_phonvec.csv"

df = pd.read_csv(DATA_PATH)

X = df[
    ["lev_norm",
     "panphon_dist",
     "phon_cosine",
     "phon_euclid"]
]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print(classification_report(y_test, pred))