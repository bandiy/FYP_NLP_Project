import numpy as np
import panphon.distance
from collections import defaultdict

# config
IPA_PATH = "data/corpora/russian2/lemma_ipa.tsv"
OUT_PATH = "data/corpora/russian2/phonetic_distance_matrix_6000.npy"

LENGTH_THRESHOLD = 2      # too long?
MAX_DISTANCE = 100.0      # value for pruned pairs


# load ipa
def load_ipa():
    print("Loading IPA...")
    ipa_list = []

    with open(IPA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            lemma, ipa = line.strip().split("\t")
            ipa_list.append(ipa)

    return ipa_list


# build matrix
def build_matrix():
    ipa_list = load_ipa()
    n = len(ipa_list)

    print(f"{n} words loaded.")

    # use float32
    matrix = np.full((n, n), MAX_DISTANCE, dtype=np.float32)

    # distance calculator
    dist = panphon.distance.Distance()

    # pre bucket
    print("Bucketing by length...")
    length_buckets = defaultdict(list)

    for idx, word in enumerate(ipa_list):
        length_buckets[len(word)].append(idx)

    print("Computing matrix...")

    for i in range(n):

        len_i = len(ipa_list[i])

        # only check
        for l in range(len_i - LENGTH_THRESHOLD, len_i + LENGTH_THRESHOLD + 1):
            if l not in length_buckets:
                continue

            for j in length_buckets[l]:
                if j < i:
                    continue  # symmetrixc

                # force first letter same pruning
                #if ipa_list[i] and ipa_list[j]:
                    #if ipa_list[i][0] != ipa_list[j][0]:
                        #continue

                d = dist.weighted_feature_edit_distance(
                    ipa_list[i],
                    ipa_list[j]
                )

                matrix[i, j] = d
                matrix[j, i] = d # symmetric

        if i % 50 == 0:
            print(f"Row {i}/{n}")

    print("Saving matrix...")
    np.save(OUT_PATH, matrix)
    print("Done.")


if __name__ == "__main__":
    build_matrix()
