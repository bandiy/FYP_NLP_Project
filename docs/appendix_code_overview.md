# Appendix — Code Repository & Script Overview

This file lists scripts grouped by version and ordered in the recommended chronological (run) order — from data extraction through preprocessing, feature extraction, experiments, and evaluation.

---

## V1

| Run Order | Script / File | Purpose |
|---:|---|---|
| 1 | src/v1/preprocessing/extract_book.py | Scrape raw text (Crime & Punishment) into corpus files |
| 2 | src/v1/preprocessing/load_text.py | Clean and tokenize raw text; save tokens |
| 3 | src/v1/preprocessing/filter_vocab.py | Build filtered vocabulary from token counts |
| 4 | src/v1/preprocessing/ipa_convert.py | Transliterate vocabulary to IPA using Epitran |
| 5 | src/v1/preprocessing/phonetic_preprocessing.py | Build IPA character vocab and fixed-length encodings |
| 6 | src/v1/embeddings/phonetic_contrastive_dataset.py | Produce triplet (anchor/positive/negative) dataset for contrastive training |
| 7 | src/v1/embeddings/train_phonetic_encoder.py | Train the LSTM-based phonetic encoder (triplet loss) |
| 8 | src/v1/embeddings/run_phonetic_embeddings.py | Generate & save phonetic embeddings using trained encoder |
| 9 | src/v1/embeddings/run_semantic_embeddings.py | Encode vocabulary with SentenceTransformer and save semantic embeddings |
| 10 | src/v1/embeddings/run_fused_embeddings.py | Fuse phonetic + semantic vectors (weighted/concatenate) and save fused embeddings |
| 11 | src/v1/visualisations/phonetic_semantic_spaces.py | t-SNE visualisations & neighbour listings for phonetic/semantic/fused spaces |
| 12 | src/v1/visualisations/phonetic_semantic_umap.py | UMAP visualisations for phonetic/semantic/fused spaces |
| 13 | src/v1/visualisations/alpha_sweep_umap.py | Sweep α (semantic↔phonetic blend) and plot trajectories |
| 14 | src/v1/visualisations/sentence_umap.py | UMAP visualisations for sentence-level embeddings |

---

## V2

| Run Order | Script / File | Purpose |
|---:|---|---|
| 1 | src/v2/preprocessing/build_lemma_list.py | Extract and filter lemma frequency list from corpora (Natasha) |
| 2 | src/v2/phonetics/build_ipa_lexicon.py | Transliterate top-N lemmas to IPA (Epitran) |
| 3 | src/v2/phonetics/build_distance_matrix.py | Compute pairwise panphon weighted feature edit-distance matrix (.npy) |
| 4 | src/v2/visualisation/build_knn_graph.py | Build k-NN phonetic graph (JSON) from distance matrix |
| 5 | src/v2/visualisation/distance_umap.py | Run UMAP on precomputed distances and plot |
| 6 | src/v2/visualisation/interactive_phonetic_explorer.py | Interactive Plotly explorer with IPA hover and neighbours; export HTML |
| 7 | src/v2/visualisation/visualise_phonetic_clusters.py | Network / community plots of phonetic clusters |
| 8 | src/v2/phonetics/test_panphon_distance.py | Panphon distance tests / dev script |

---

## semanticsv2

| Run Order | Script / File | Purpose |
|---:|---|---|
| 1 | src/semanticsv2/!test_load.py | Quick loader / sanity check for lemma filtering |
| 2 | src/semanticsv2/build_embeddings.py | Encode lemma list with SentenceTransformer; save embeddings and lemma list |
| 3 | src/semanticsv2/build_translations.py | Optional RU→EN translations of lemmas using a translation pipeline |
| 4 | src/semanticsv2/cluster_embeddings.py | PCA + KMeans clustering on semantic embeddings |
| 5 | src/semanticsv2/reduce_dimensions.py | PCA → UMAP pipeline and plot/save semantic projection |
| 6 | src/semanticsv2/nearest_neighbours.py | CLI tool for nearest-neighbour lookup using saved embeddings and translations |

---

## analysisv2

| Run Order | Script / File | Purpose |
|---:|---|---|
| 1 | src/analysisv2/global_correlation.py | Compute Spearman/Pearson correlation between phonetic & semantic distance matrices |
| 2 | src/analysisv2/validate_global_correlation.py | Permutation test and bootstrap CI for the global correlation statistic |
| 3 | src/analysisv2/compare_spaces.py | Compute neighbour overlaps & rank agreement between phonetic and semantic KNNs |
| 4 | src/analysisv2/fused_space_experiment.py | Fuse semantic distances and phonetic matrix for chosen α and evaluate metrics |
| 5 | src/analysisv2/plot_fusion_curve.py | Sweep α and produce an interactive Plotly fusion-curve HTML |
| 6 | src/analysisv2/umap_fused_graph.py | UMAP projections of fused distance matrices for selected α values |

---

## V3 (russian_french)

| Run Order | Script / File | Purpose |
|---:|---|---|
| 1 | src/v3_french_russian/preprocessing/get_dostoyevsky.py | Scrape Russian corpora (Dostoevsky) into corpus folder |
| 2 | src/v3_french_russian/preprocessing/get_hugo.py | Download & clean French Gutenberg texts |
| 3 | src/v3_french_russian/preprocessing/build_russian_lemmas.py | Lemma extraction & frequency list for Russian (Natasha) |
| 4 | src/v3_french_russian/preprocessing/build_french_lemmas.py | Lemma extraction & frequency list for French (spaCy) |
| 5 | src/v3_french_russian/preprocessing/build_russian_ipa.py | Transliterate Russian lemmas → IPA (Epitran) |
| 6 | src/v3_french_russian/preprocessing/build_french_ipa.py | Transliterate French lemmas → IPA (Epitran) |
| 7 | src/v3_french_russian/embeddings/build_joint_semantic_embeddings.py | Encode RU+FR lemma lists with SentenceTransformer and save embeddings |
| 8 | src/v3_french_russian/embeddings/cross_semantic_neighbours.py | Retrieve top-FR semantic neighbours for Russian targets |
| 9 | src/v3_french_russian/embeddings/compare_spaces.py | Vocabulary-wide phonetic vs semantic neighbour overlap analysis |
| 10 | src/v3_french_russian/embeddings/local_a_sweep.py | Local α-sweep on shortlist to find best α per word set |
| 11 | src/v3_french_russian/embeddings/full_vocab_hybrid_test.py | Full-vocab α sweep reporting MRR across gold pairs |
| 12 | src/v3_french_russian/embeddings/full_vocab_bootstrap.py | Bootstrap test of hybrid vs semantic MRR improvements |
| 13 | src/v3_french_russian/embeddings/local_bootstrap_hybrid_vs_semantic.py | Local bootstrap comparison on shortlist |
| 14 | src/v3_french_russian/embeddings/local_per_word_affect_analysis.py | Per-word Δ MRR analysis on shortlist |
| 15 | src/v3_french_russian/embeddings/full_vocab_per_word_affect_analysis.py | Per-word Δ MRR analysis across full gold set |
| 16 | src/v3_french_russian/embeddings/validate_gold.py | Sanity checks and rank reports for curated gold set |

---

## V4

| Run Order | Script / File | Purpose |
|---:|---|---|
| 1 | src/v4/2_build_7lang_pairs.py | Use LingPy to cluster forms and produce pairwise cognate/non-cognate CSV for 7 languages |
| 2 | src/v4/7_semantic_negatives_added.py | Generate cross-concept semantic negatives and augment dataset |
| 3 | src/v4/v6_hard_negatives.py | (Optional) Generate hard negatives by selecting semantically-close non-positives |
| 4 | src/v4/3_extract_features.py | Compute phonetic features (panphon, Levenshtein, phonetic vectors) for each pair |
| 5 | src/v4/5_extract_features_with_semantics.py | Add semantic similarity (SentenceTransformer) to feature set |
| 6 | src/v4/4_train_classifier.py | Train phonetic-only classifier and report metrics |
| 7 | src/v4/6_train_combined_classifier.py | Train phonetic-only, semantic-only, and hybrid classifiers; report feature importances |
| 8 | src/v4/eval_gold_hybrid.py | Evaluate trained models on curated RU–FR gold pairs (compare probabilities) |
| 9 | src/v4/lexstattest.py | LingPy LexStat threshold diagnostics and RF pair counts |
