# Final V4 Business Entity Resolution Submission

This package generates the challenge test predictions from the completed V4 XGBoost experiment. It does not retrain or alter V4. It uses only the provided challenge files and the saved V4 model artifacts.

## Environment and files

Use Python 3.9 or newer. Install the dependencies from the challenge project root:

```bash
python3 -m pip install -r code/business_entity_resolution/requirements.txt
```

The project root must contain:

```text
dataset/
  test/test_source1.tsv
  test/test_source2.tsv
  test/test_source3.tsv
code/business_entity_resolution/artifacts/v4/cap_100/
  v4_xgboost_india.json
  v4_xgboost_us.json
  v4_validation_report.json
code/business_entity_resolution/scripts/predict_v4.py
utils/validate_submission.py
```

The trained V4 models and validation report are submission artifacts. The validation cache is not required for inference and is not included. The model files were trained with the V4 settings documented below. They are loaded directly; the final inference command does not train or tune a model.

## Generate final outputs

Run from the clean project root (or the extracted final ZIP) to write the required `output/` files:

```bash
python3 code/business_entity_resolution/scripts/predict_v4.py \
  --project-root . \
  --max-candidates-per-entity 100 \
  --output-dir output
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, plus a small inference report at `code/business_entity_resolution/artifacts/v4/cap_100/v4_test_inference_report.json`. In a development workspace that still contains earlier baseline files under `output/`, set `--output-dir final_submission/output` to preserve them; use that same directory in the validator command.

Validate the generated outputs with the supplied challenge validator:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

`--check-ids` additionally confirms that every candidate and match exists in the test Source 2/3 files. It uses more memory than the validator's default mode.

## Input schema and candidate generation

The source TSV schema is `entity_id`, `business_name`, `business_address`, and `country`. Ground truth is not used during inference. There are no separate city, state, postal-code, telephone, or email columns. Existing V2/V3 logic may conservatively extract phone, email, and postal values embedded in names or addresses.

V4 partitions the target index by the normalized, open-set country string and indexes only test Source 2 and Source 3 records. It retains the V3/V2 top-50 candidate seed, then unions the V3-ranked candidates available up to the selected cap with three bounded proposal routes:

1. Equivalent terminal legal-form name probes through the existing exact-name index.
2. A single shared three-character name-prefix proposal for postings with at most 50 records.
3. Four-character prefixes of long normalized address tokens, with at most 25 target records per prefix.

Proposals are deduplicated by target `entity_id`, deterministically ranked by blocking evidence, and capped at 100 per Source 1 entity. This uses inverted postings and bounded proposals; it does not construct a Source 1 × Source 2/3 Cartesian product.

`candidate_pairs.tsv` records the final post-union, post-deduplication, post-cap candidate IDs. Those same IDs, in the same order, are converted to feature vectors and scored. A predicted match is selected only from this recorded candidate list.

## V4 features and model

V4 keeps the existing 29 V3/V2 features unchanged. They comprise the 19 baseline signals below plus 10 conservative V2 signals:

- Name: exact normalized name, character trigram similarity, token Jaccard and containment, character bigram Jaccard, length ratio, prefix ratio, legal-suffix-stripped core equality, three-character token-prefix overlap, and character four-gram Jaccard.
- Address: exact normalized address, character trigram similarity, token Jaccard and containment, length ratio, numeric-token overlap, and character four-gram Jaccard.
- Pair/source: country equality, Source 3 indicator, missing-name and missing-address indicators, combined name/address similarity mean and product.
- Embedded identifiers: exact/conflict signals for email, phone, and postal code when conservative extraction finds a value.

Normalization reuses Unicode NFKC/case-folding, punctuation and whitespace cleanup, ampersand expansion, conservative legal-form canonicalization, address abbreviations and compass directions. It does not merge records on a name or address alone; the classifier scores the field-specific similarities together.

The saved scorer is an XGBoost binary classifier with 300 estimators, learning rate 0.05, depth 5, minimum child weight 3, subsample 0.8, column sample 0.8, L1 regularization 0.1, L2 regularization 1.0, seed 1729, `n_jobs=-1`, and histogram tree construction. A depth-5 tree has at most 63 nodes, so the 300-tree ensemble has at most 18,900 nodes and is comfortably within the challenge's 8-billion-parameter ceiling. XGBoost is Apache License 2.0 software. Training used positive sample weight 2.0, all captured positives, and deterministic per-entity negative sampling (at least 20 negatives, or up to five per positive). Model fitting and validation used 1,000 stable-hash buckets: buckets 10–69 for training and 0–9 for validation, with at most 10,000 training and 2,000 validation S1 entities per trained country.

## Thresholds and unseen countries

The V4 validation report selects a threshold independently for each trained country by maximum validation macro F0.5 over the recorded threshold sweep, preferring the higher threshold in a tie. The completed V4 run selected 0.85 for India and 0.85 for the US.

Country labels remain open-set strings. A test country with a trained V4 model uses that model and its validation threshold. An unseen country such as France is retained and processed using the arithmetic mean of the saved trained V4 models' predicted probabilities; its threshold is the arithmetic mean of their validation-selected thresholds. With the completed run's India and US thresholds, the fallback threshold is 0.85. No France-specific model is fabricated or trained.

## Validation and fair play

V4's measured validation results from the completed cap-100 run are recorded in `artifacts/v4/cap_100/v4_validation_report.json`. The methodology document in the submission package includes the overall and per-country validation values from that report and the final test-output counts from `v4_test_inference_report.json`. Test labels are not available, so final test output counts are not accuracy estimates.

No external lookup, API, geocoding, business database, or external data augmentation is used. XGBoost is distributed under Apache License 2.0; the package contains the trained model files and inference source needed to reproduce test outputs from the supplied challenge data.

## Output format

- `matching_results.tsv`: one row per test Source 1 entity; columns `source1_entity_id` and comma-separated `matched_entity_ids`. An entity with no predictions has an empty second field.
- `candidate_pairs.tsv`: one row per test Source 1 entity; columns `source1_entity_id` and comma-separated final `candidate_entity_ids`. Empty candidate sets have an empty second field.

Both are UTF-8 tab-separated files. Source 1 rows are emitted once across every observed country, including France and any other unseen country.
