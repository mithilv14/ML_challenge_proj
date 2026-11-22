# ML Challenge 2026 — Business Entity Resolution Methodology

**Team Name:** Blackout   
**Submission Date:** 27/09/2026

---

## 1. Executive Summary

The submission uses the completed V4 candidate-generation and pair-scoring experiment. It compares each test Source 1 record with a bounded set of Source 2/Source 3 records from the same country, then scores the resulting pairs with the saved V4 XGBoost model. V4 validation achieved overall blocking recall 0.8661264879, precision 0.9560420004, recall 0.7704001147, and macro F0.5 0.8699103668 with a cap of 100 candidates per Source 1 record.

## 2. Problem Formulation and Dataset

This is multi-match entity resolution. For every Source 1 business, the system predicts zero or more corresponding Source 2/Source 3 entity IDs. Inputs are the supplied tab-separated challenge files only.

Each source file has fields `entity_id`, `business_name`, `business_address`, and `country`; training labels have `source1_entity_id` and comma-separated `matched_entity_ids`. There are no dedicated city, state, postal-code, telephone, or email columns. Some phone, email, and postal values can be conservatively extracted from name/address text. France is present in test data but was not a training country.

## 3. Candidate Generation and Blocking

V4 retains the V3/V2 country-partitioned target blocker as its seed. It indexes only Source 2 and Source 3 records, then for each Source 1 record combines:

1. The existing V3/V2 candidate seed, retaining its top 50 candidate identities.
2. V3-ranked blocker candidates up to the selected V4 cap.
3. Normalized exact-name probes for equivalent terminal legal-form spellings.
4. A bounded single three-character name-token-prefix proposal for postings of up to 50 targets.
5. Bounded four-character prefixes of long normalized address tokens, with no more than 25 targets per prefix.

Candidate proposals are deduplicated by the target record's `entity_id`, deterministically ranked using blocking evidence, and capped at 100 per Source 1 record. The inverted indexes and posting limits avoid a Cartesian product over the source tables. `candidate_pairs.tsv` contains the final candidate set after union, deduplication, ranking, and cap; exactly that ordered set is featurized and scored.

## 4. Normalization, Features, and Model

Normalization reuses Unicode NFKC, case folding, punctuation and whitespace cleanup, ampersand expansion, conservative business legal-form handling, common address abbreviations, and compass-direction expansion. It preserves the original record identity and does not make fuzzy normalization decisions that merge entities.

The model uses the unchanged 29 V3/V2 features:

- 19 baseline fields: exact normalized name; name character-trigram, token Jaccard, token containment, character-bigram, length-ratio and prefix similarities; exact normalized address; address character-trigram, token Jaccard, token containment, length-ratio and numeric-token-overlap similarities; country equality; Source 3 indicator; name/address missingness; and combined name/address similarity mean and product.
- 10 V2 fields: legal-suffix-stripped name-core equality; name token-prefix overlap; name and address character four-gram Jaccard; and exact/conflict flags for embedded email, phone, and postal values.

The scorer is an XGBoost binary classifier with 300 estimators, learning rate 0.05, maximum depth 5, minimum child weight 3, subsample 0.8, column sample 0.8, L1 regularization 0.1, L2 regularization 1.0, random seed 1729, all available CPU workers, and histogram tree construction. A depth-5 tree has at most 63 nodes, so the 300-tree ensemble has at most 18,900 nodes and is comfortably within the challenge's 8-billion-parameter ceiling. XGBoost is Apache License 2.0 software. Positive examples have weight 2.0. Training keeps all captured positive candidate pairs and selects negatives deterministically per S1 (at least 20, or up to five per positive). A 1,000-bucket stable hash split assigns buckets 10–69 to training and 0–9 to validation, with up to 10,000 training and 2,000 validation S1 rows per country.

## 5. Threshold Selection and Unseen Countries

For each trained country, validation evaluates thresholds from 0.50 through 0.95 and selects the threshold with maximum macro F0.5; ties prefer the higher threshold. The completed V4 run selected 0.85 for India and 0.85 for the US.

Country is an open-set string. For a known country, inference uses its saved V4 model and selected threshold. For an unseen country such as France, inference averages probabilities from all saved trained-country V4 models and uses the arithmetic mean of their validation-selected thresholds. For the current India/US models this gives a fallback threshold of 0.85. France is neither dropped nor assigned a fabricated country-specific model.

## 6. Validation Results

<!-- GENERATED_V4_VALIDATION_RESULTS_BEGIN -->

| Scope | Blocking recall | Precision | Recall | Macro F0.5 | Threshold | Candidate pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall, country-tuned | 0.8661264879 | 0.9560420004 | 0.7704001147 | 0.8699103668 | per-country best | 201,947 |

| Country | Blocking recall | Precision | Recall | Macro F0.5 | Best threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| India | 0.8556581986 | 0.9279456463 | 0.7491339492 | 0.8390016333 | 0.85 |
| US | 0.8764605301 | 0.9838795394 | 0.7913935594 | 0.9008191004 | 0.85 |

<!-- GENERATED_V4_VALIDATION_RESULTS_END -->

V4 validation metrics are distinct from final test output counts. Test ground truth is unavailable, so the final predictions are not presented as test accuracy.

## 7. Final Test Inference Results

<!-- GENERATED_FINAL_TEST_RESULTS -->

## 8. Final Output Files

- `output/matching_results.tsv`: exactly one row per test Source 1 record, with the comma-separated predicted Source 2/Source 3 IDs or an empty field.
- `output/candidate_pairs.tsv`: exactly one row per test Source 1 record, with the actual post-cap candidate IDs passed to the classifier or an empty field.

The above files are uploaded in the following google drive link:
https://drive.google.com/drive/folders/1HA0acU9XG4DaUnulwXQKzvKsuC1pSvNa?usp=sharing

The inference package verifies the candidate cap and per-row deduplication. The official validator checks Source 1 coverage and uniqueness, ID prefixes, candidate/match subset integrity, and—when run with `--check-ids`—candidate and match existence in test Source 2/Source 3.

## 9. Fair Play and Reproducibility

No external data, web search, API, business database, geocoding, or third-party lookup was used. All normalization, blocking, features, training labels, models, and predictions use the supplied challenge files. XGBoost is Apache License 2.0 software; the included JSON models are the trained V4 artifacts.

From the challenge project root, install the dependencies and run:

```bash
python3 -m pip install -r code/business_entity_resolution/requirements.txt
python3 code/business_entity_resolution/scripts/predict_v4.py \
  --project-root . \
  --max-candidates-per-entity 100 \
  --output-dir output
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

The trained model files and V4 validation report must be present under `code/business_entity_resolution/artifacts/v4/cap_100/`. The model artifacts are included in the final ZIP; the large V2/V4 validation caches are not required for inference and are excluded.
