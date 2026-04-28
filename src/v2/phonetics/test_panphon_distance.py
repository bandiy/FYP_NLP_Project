#TEST ONLY
import panphon.distance as pd

# initialise distance calculator
dist = pd.Distance()

def test_distances():
    word1 = "ɡovorʲitʲ"   # говорить
    word2 = "ɡovorʲit"    # говорить without final soft
    word3 = "dom"         # дом
    word4 = "ɡorod"       # город

    print("Feature edit distance:")
    print("w1 vs w2:", dist.feature_edit_distance(word1, word2))
    print("w1 vs w3:", dist.feature_edit_distance(word1, word3))
    print("w3 vs w4:", dist.feature_edit_distance(word3, word4))

    print("\nWeighted feature edit distance:")
    print("w1 vs w2:", dist.weighted_feature_edit_distance(word1, word2))
    print("w1 vs w3:", dist.weighted_feature_edit_distance(word1, word3))
    print("w3 vs w4:", dist.weighted_feature_edit_distance(word3, word4))

if __name__ == "__main__":
    test_distances()
