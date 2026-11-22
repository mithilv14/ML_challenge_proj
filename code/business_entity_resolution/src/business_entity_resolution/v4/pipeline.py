import array
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from time import perf_counter

from ..data import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, iter_source, load_ground_truth
from ..evaluate import ValidationAccumulator
from ..normalize import normalize_country
from ..pipeline import _sample_records
from ..v2.evaluate import tuned_overall
from ..v2.features import V2_FEATURE_NAMES, pair_features_v2, prepare_source1_v2
from .blocking import V4TargetIndex
from .config import (
    POSITIVE_SAMPLE_WEIGHT,
    THRESHOLDS,
    V3_CANDIDATE_CAP,
    V3_REPORTED_REFERENCE,
    V4Config,
    XGBOOST_PARAMS,
)


def _stable_hash(value):
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def inspect_dataset_schema(project_root):
    """Read only TSV headers and confirm the real challenge schema used by V4."""
    root = Path(project_root)
    headers = {}
    for split in ("train", "test"):
        for source in ("source1", "source2", "source3"):
            path = root / "dataset" / split / f"{split}_{source}.tsv"
            if not path.is_file():
                if split == "test":
                    continue
                raise FileNotFoundError(f"Required dataset source file is missing: {path}")
            with path.open("r", encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle, delimiter="\t"), [])
            if tuple(header) != SOURCE_COLUMNS:
                raise ValueError(
                    f"Unexpected columns in {path}: {header!r}; expected {list(SOURCE_COLUMNS)!r}"
                )
            headers[f"{split}_{source}"] = header

    truth_path = root / "dataset" / "train" / "train_ground_truth.tsv"
    if not truth_path.is_file():
        raise FileNotFoundError(f"Required ground-truth file is missing: {truth_path}")
    with truth_path.open("r", encoding="utf-8", newline="") as handle:
        truth_header = next(csv.reader(handle, delimiter="\t"), [])
    if tuple(truth_header) != GROUND_TRUTH_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {truth_path}: {truth_header!r}; "
            f"expected {list(GROUND_TRUTH_COLUMNS)!r}"
        )
    headers["train_ground_truth"] = truth_header
    return {
        "headers": headers,
        "source_fields": list(SOURCE_COLUMNS),
        "ground_truth_fields": list(GROUND_TRUTH_COLUMNS),
        "separate_city_state_postal_phone_email_columns": [],
        "note": (
            "Blocking may use entity_id, business_name, business_address, and country. "
            "Postal/phone/email values are available only when conservatively extracted "
            "from the name/address text by the existing V3/V2 code."
        ),
    }


def _load_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required for the V4 experiment. Install the V3 XGBoost dependency "
            "in this environment; V4 does not substitute another scorer."
        ) from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("V4 requires NumPy, normally installed with XGBoost.") from exc
    return XGBClassifier, np


def _append_training_pair(flat_features, labels, weights, features, label):
    if len(features) != len(V2_FEATURE_NAMES) or not all(
        math.isfinite(float(value)) for value in features
    ):
        raise ValueError("V4 received an invalid vector from the unchanged V3 29-feature pipeline")
    flat_features.extend(float(value) for value in features)
    labels.append(int(label))
    weights.append(POSITIVE_SAMPLE_WEIGHT if label else 1.0)


def _fit_country(train_records, labels_by_s1, index, config, XGBClassifier, np):
    if array.array("f").itemsize != 4 or array.array("i").itemsize != 4:
        raise RuntimeError("V4 requires 32-bit C float and integer arrays for compact matrices")
    flat_features = array.array("f")
    label_values = array.array("i")
    sample_weights = array.array("f")
    candidate_pairs = positive_pairs = negative_pairs = 0

    for position, source1 in enumerate(train_records, start=1):
        truth = labels_by_s1.get(source1.entity_id, set())
        result = index.candidates_with_sources(source1)
        candidate_indices = result["candidate_indices"]
        candidate_ids = result["candidate_ids"]
        candidate_pairs += len(candidate_ids)
        positives, negatives = [], []
        for idx, entity_id in zip(candidate_indices, candidate_ids):
            (positives if entity_id in truth else negatives).append(idx)

        negative_count = min(
            len(negatives),
            max(
                config.negative_samples_per_entity,
                len(positives) * config.negative_to_positive_ratio,
            ),
        )
        rng = random.Random(_stable_hash(source1.entity_id) ^ config.seed)
        selected = [(idx, 1) for idx in positives]
        selected.extend(
            (idx, 0) for idx in (rng.sample(negatives, negative_count) if negative_count else [])
        )
        rng.shuffle(selected)
        prepared = prepare_source1_v2(source1)
        for idx, label in selected:
            _append_training_pair(
                flat_features,
                label_values,
                sample_weights,
                pair_features_v2(prepared, index.feature_record(idx)),
                label,
            )
            positive_pairs += label
            negative_pairs += 1 - label
        if position % 1000 == 0:
            print(f"  V4 prepared {position:,}/{len(train_records):,} {index.country} S1 rows", flush=True)

    row_count = len(label_values)
    if not row_count or not positive_pairs or not negative_pairs:
        raise ValueError(
            f"Insufficient V4 training pairs for {index.country}: "
            f"positive={positive_pairs}, negative={negative_pairs}"
        )
    X = np.frombuffer(flat_features, dtype=np.float32).reshape(row_count, len(V2_FEATURE_NAMES))
    y = np.frombuffer(label_values, dtype=np.int32)
    weights = np.frombuffer(sample_weights, dtype=np.float32)
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model, {
        "training_candidate_pairs": candidate_pairs,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "training_examples": row_count,
        "positive_sample_weight": POSITIVE_SAMPLE_WEIGHT,
    }


def _sum_generation_stats(total, row_stats):
    for key in (
        "v3_candidates_at_50",
        "v3_candidates_available_to_cap",
        "v4_unique_candidates_before_cap",
        "v4_candidates_after_cap",
        "v4_candidates_added_beyond_v3_50",
        "candidate_proposals_deduplicated",
        "candidate_cap_applied",
    ):
        total[key] = total.get(key, 0) + row_stats[key]
    for strategy, count in row_stats["strategy_unique_candidate_counts"].items():
        total.setdefault("strategy_unique_candidate_counts", {})[strategy] = (
            total.setdefault("strategy_unique_candidate_counts", {}).get(strategy, 0) + count
        )


def _country_cache_rows(cache, country, validation_records, target_count):
    if cache is None:
        return None
    item = cache.get("countries", {}).get(country)
    if item is None:
        raise ValueError(f"Existing V2 validation cache has no {country!r} rows")
    if int(item.get("target_entities", -1)) != target_count:
        raise ValueError(f"V2 cache target count does not match the rebuilt {country} target index")
    rows = item.get("rows", [])
    expected_ids = [record.entity_id for record in validation_records]
    if [row.get("s1_id") for row in rows] != expected_ids:
        raise ValueError(f"V2 cache S1 validation sample differs from the V4 split for {country}")
    return rows


def _overall_threshold_sweep(country_reports, thresholds):
    result = {}
    for threshold in thresholds:
        key = str(threshold)
        entities = sum(report["v4"]["entities"] for report in country_reports.values())
        true_matches = sum(report["v4"]["true_matches"] for report in country_reports.values())
        predicted = sum(
            report["v4"]["thresholds"][key]["predicted_pairs"]
            for report in country_reports.values()
        )
        true_positive = sum(
            round(
                report["v4"]["thresholds"][key]["precision_micro"]
                * report["v4"]["thresholds"][key]["predicted_pairs"]
            )
            for report in country_reports.values()
        )
        macro = sum(
            report["v4"]["thresholds"][key]["macro_f0_5"]
            * report["v4"]["entities"]
            for report in country_reports.values()
        ) / entities if entities else 0.0
        result[key] = {
            "precision_micro": true_positive / predicted if predicted else 0.0,
            "recall_micro": true_positive / true_matches if true_matches else 0.0,
            "macro_f0_5": macro,
            "predicted_pairs": predicted,
            "validation_entities": entities,
        }
    return result


def train_v4_pipeline(config):
    """Train/evaluate V4 with the V3 XGBoost settings and a separate V4 candidate union."""
    XGBClassifier, np = _load_xgboost()
    started = perf_counter()
    schema = inspect_dataset_schema(config.project_root)
    train_dir = config.data_dir / "train"
    source1_path = train_dir / "train_source1.tsv"
    source2_path = train_dir / "train_source2.tsv"
    source3_path = train_dir / "train_source3.tsv"
    truth_path = train_dir / "train_ground_truth.tsv"
    cache_path = config.base_artifacts_dir / "v2_validation_cache.json"
    v2_cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else None

    countries = sorted({
        normalize_country(record.country)
        for record in iter_source(source1_path, "S1")
        if normalize_country(record.country)
    })
    country_reports = {}
    output_cache = {}
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    for country in countries:
        print(f"Preparing V4 samples for country={country}", flush=True)
        train_records, validation_records = _sample_records(source1_path, country, config)
        train_ids = {record.entity_id for record in train_records}
        validation_ids = {record.entity_id for record in validation_records}
        if train_ids & validation_ids:
            raise ValueError(f"V4 train/validation split overlaps for {country}")
        train_labels = load_ground_truth(truth_path, train_ids)
        if train_ids - set(train_labels):
            raise ValueError(f"Ground truth is missing sampled V4 training rows for {country}")

        index = V4TargetIndex(
            country,
            max_token_posting=config.max_token_posting,
            max_candidates=config.max_candidates_per_entity,
            v3_candidate_cap=V3_CANDIDATE_CAP,
        ).extend_from_files(source2_path, source3_path)
        print(f"Indexed {len(index):,} S2/S3 records for V4 country={country}", flush=True)

        model, training_stats = _fit_country(
            train_records, train_labels, index, config, XGBClassifier, np
        )
        model_path = config.artifacts_dir / f"v4_xgboost_{country}.json"
        model.save_model(str(model_path))

        cached_rows = _country_cache_rows(v2_cache, country, validation_records, len(index))
        if cached_rows is None:
            validation_labels = load_ground_truth(truth_path, validation_ids)
            if validation_ids - set(validation_labels):
                raise ValueError(f"Ground truth is missing sampled V4 validation rows for {country}")
        else:
            validation_labels = None

        prepared_rows = []
        flat_features = array.array("f")
        generation_totals = {}
        v3_blocking = ValidationAccumulator(config.thresholds)
        for position, record in enumerate(validation_records, start=1):
            generated = index.candidates_with_sources(record)
            stats = generated["stats"]
            _sum_generation_stats(generation_totals, stats)
            if cached_rows is not None:
                cached_v3_ids = cached_rows[position - 1].get("v2_candidate_ids", [])
                if cached_v3_ids != generated["v3_candidate_ids"]:
                    raise ValueError(
                        f"V4's retained V3 candidates differ from the saved V2/V3 cache at "
                        f"{record.entity_id}; check the V3 blocker and candidate cap"
                    )
                truth_ids = cached_rows[position - 1].get("truth_ids", [])
            else:
                truth_ids = sorted(validation_labels.get(record.entity_id, set()))

            v3_ids = generated["v3_candidate_ids"]
            v3_blocking.add(truth_ids, v3_ids, [0.0] * len(v3_ids))
            candidate_ids = generated["candidate_ids"]
            candidate_indices = generated["candidate_indices"]
            prepared = prepare_source1_v2(record)
            for idx in candidate_indices:
                feature_vector = pair_features_v2(prepared, index.feature_record(idx))
                if len(feature_vector) != len(V2_FEATURE_NAMES) or not all(
                    math.isfinite(float(value)) for value in feature_vector
                ):
                    raise ValueError("V4 candidate did not produce a finite 29-feature V3 vector")
                flat_features.extend(float(value) for value in feature_vector)
            prepared_rows.append({
                "s1_id": record.entity_id,
                "truth_ids": truth_ids,
                "v3_candidate_ids": v3_ids,
                "candidate_ids": candidate_ids,
                "candidate_count": len(candidate_indices),
                "stats": stats,
            })
            if position % 500 == 0:
                print(
                    f"  V4 prepared validation {position:,}/{len(validation_records):,} "
                    f"{country} S1 rows",
                    flush=True,
                )

        pair_count = sum(row["candidate_count"] for row in prepared_rows)
        if pair_count:
            X_val = np.frombuffer(flat_features, dtype=np.float32).reshape(
                pair_count, len(V2_FEATURE_NAMES)
            )
            all_scores = model.predict_proba(X_val)[:, 1].astype(float).tolist()
        else:
            all_scores = []
        score_offset = 0
        v4_accumulator = ValidationAccumulator(config.thresholds)
        scored_rows = []
        for row in prepared_rows:
            count = row["candidate_count"]
            scores = all_scores[score_offset:score_offset + count]
            score_offset += count
            v4_accumulator.add(row["truth_ids"], row["candidate_ids"], scores)
            scored_rows.append({
                "s1_id": row["s1_id"],
                "truth_ids": row["truth_ids"],
                "v3_candidate_ids": row["v3_candidate_ids"],
                "candidate_ids": row["candidate_ids"],
                "v4_scores": scores,
            })
        v4_metrics = v4_accumulator.report(target_count=len(index))
        v3_metrics = v3_blocking.report(target_count=len(index))
        country_reports[country] = {
            "country": country,
            "train_entities": len(train_records),
            "training": training_stats,
            "target_entities": len(index),
            "v3_blocking": {
                "blocking_recall": v3_metrics["blocking_recall"],
                "candidate_pairs": v3_metrics["candidate_pairs"],
                "blocked_true_matches": v3_metrics["blocked_true_matches"],
                "true_matches": v3_metrics["true_matches"],
                "average_candidates_per_entity": v3_metrics["average_candidates_per_entity"],
                "maximum_candidates_per_entity": v3_metrics["maximum_candidates_per_entity"],
            },
            "v4": v4_metrics,
            "candidate_generation": generation_totals,
            "model_path": str(model_path),
            "validation_rows": scored_rows,
        }
        output_cache[country] = {
            "target_entities": len(index),
            "validation_rows": scored_rows,
        }
        print(
            f"V4 validated {country}: blocking_recall={v4_metrics['blocking_recall']:.4f} "
            f"precision={v4_metrics['precision_micro']:.4f} "
            f"recall={v4_metrics['recall_micro']:.4f} "
            f"macro_f0.5={v4_metrics['macro_f0_5']:.4f} "
            f"candidates={v4_metrics['candidate_pairs']:,}",
            flush=True,
        )

    v4_overall = tuned_overall({
        country: detail["v4"] for country, detail in country_reports.items()
    })
    v3_blocked = sum(
        detail["v3_blocking"]["blocked_true_matches"]
        for detail in country_reports.values()
    )
    v3_true = sum(
        detail["v3_blocking"]["true_matches"]
        for detail in country_reports.values()
    )
    v3_candidate_total = sum(
        detail["v3_blocking"]["candidate_pairs"] for detail in country_reports.values()
    )
    v3_blocking_overall = {
        "blocking_recall": v3_blocked / v3_true if v3_true else 0.0,
        "candidate_pairs": v3_candidate_total,
        "validation_entities": sum(detail["v4"]["entities"] for detail in country_reports.values()),
    }
    report = {
        "version": "V4 candidate-recall experiment with unchanged V3 XGBoost and 29 features",
        "runtime_seconds": round(perf_counter() - started, 2),
        "runtime_minutes": round((perf_counter() - started) / 60.0, 2),
        "dataset_schema": schema,
        "v3_reference_reported_by_user": V3_REPORTED_REFERENCE,
        "v3_blocking_measured_on_v4_validation_sample": v3_blocking_overall,
        "v4": v4_overall,
        "overall_threshold_sweep": _overall_threshold_sweep(country_reports, config.thresholds),
        "thresholds": list(config.thresholds),
        "threshold_selection": "Per-country validation macro F0.5 maximum; ties select the higher threshold.",
        "xgboost_params_reused_from_v3": XGBOOST_PARAMS,
        "positive_sample_weight": POSITIVE_SAMPLE_WEIGHT,
        "feature_names_reused_from_v3": list(V2_FEATURE_NAMES),
        "candidate_cap": config.max_candidates_per_entity,
        "v3_candidate_cap_preserved_as_seed": V3_CANDIDATE_CAP,
        "sampling": {
            "hash_buckets": config.hash_buckets,
            "train_hash_range": list(config.train_hash_range),
            "validation_hash_range": list(config.validation_hash_range),
            "max_train_per_country": config.max_train_per_country,
            "max_validation_per_country": config.max_validation_per_country,
            "negative_samples_per_entity": config.negative_samples_per_entity,
            "negative_to_positive_ratio": config.negative_to_positive_ratio,
            "seed": config.seed,
        },
        "country_reports": {
            country: {
                key: value for key, value in detail.items() if key != "validation_rows"
            }
            for country, detail in country_reports.items()
        },
        "notes": [
            "V4 retains the V3/V2 candidate set at cap 50 as an unpruned seed, then unions V3 cap expansion and new bounded blockers before final pruning.",
            "Candidate identity is entity_id, the dataset's Source 2/Source 3 record key; duplicate IDs from different proposal paths are emitted once.",
            "The inspected schema has no separate city, state, postal, phone, or email columns; only name/address/country and ID are used.",
            "The original V3 model files and artifacts are read-only and are not used as V4 output destinations.",
        ],
    }
    (config.artifacts_dir / "v4_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cache_artifact = {
        "version": "V4 validation scores and candidate sets",
        "candidate_cap": config.max_candidates_per_entity,
        "thresholds": list(config.thresholds),
        "countries": output_cache,
    }
    (config.artifacts_dir / "v4_validation_cache.json").write_text(
        json.dumps(cache_artifact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    return report


def run_smoke_test(project_root):
    """Tiny local blocker/feature/path check; never trains or builds dataset indexes."""
    from ..data import Record

    schema = inspect_dataset_schema(project_root)
    if schema["source_fields"] != ["entity_id", "business_name", "business_address", "country"]:
        raise AssertionError("Unexpected real dataset schema in V4 smoke check")
    if len(V2_FEATURE_NAMES) != 29:
        raise AssertionError("V3 feature schema is no longer 29 fields")

    # Legal-form alias recovery without allocating a target index over real data.
    alias_index = V4TargetIndex("us", max_candidates=75)
    alias_target = Record(
        "S2-alias", "Harbor Limited Liability Co", "Harbor Lane", "US", "S2"
    )
    alias_index.add(alias_target)
    for number in range(110):
        alias_index.add(Record(
            f"S2-harbor-{number:03d}", f"Harbor Other Entity {number}",
            f"Unrelated Road {number}", "US", "S2",
        ))
    alias_source = Record("S1-alias", "Harbor LLC", "Remote Place", "US", "S1")
    alias_result = alias_index.candidates_with_sources(alias_source)
    if alias_target.entity_id not in alias_result["candidate_ids"]:
        raise AssertionError("Equivalent terminal legal-form blocker did not recover its target")
    if alias_target.entity_id in alias_result["v3_candidate_ids"]:
        raise AssertionError("Legal-form smoke pair was already a V3 seed; fixture is not isolating V4")

    # A bounded single-token name prefix proposal whose posting is too common for V3.
    prefix_index = V4TargetIndex("us", max_candidates=75)
    prefix_target = Record(
        "S2-prefix-target", "Nortstar Advisory", "Remote Place", "US", "S2"
    )
    prefix_index.add(prefix_target)
    for number in range(8):
        prefix_index.add(Record(
            f"S2-north-prefix-{number}", f"Norwell Holdings {number}", f"Elsewhere {number}", "US", "S2"
        ))
    prefix_source = Record("S1-prefix", "Northstar Consulting", "Far Away", "US", "S1")
    prefix_result = prefix_index.candidates_with_sources(prefix_source)
    if prefix_target.entity_id not in prefix_result["candidate_ids"]:
        raise AssertionError("Relaxed single name-prefix blocker failed")
    if prefix_target.entity_id in prefix_result["v3_candidate_ids"]:
        raise AssertionError("Prefix smoke pair was already a V3 seed; fixture is not isolating V4")

    # A rare address prefix recovers a one-character address token variation.
    address_index = V4TargetIndex("us", max_candidates=75)
    address_target = Record(
        "S2-address-target", "Unrelated Target Company", "Mercer", "US", "S2"
    )
    address_index.add(address_target)
    address_source = Record("S1-address", "Another Source Company", "Mercar", "US", "S1")
    address_result = address_index.candidates_with_sources(address_source)
    if address_target.entity_id not in address_result["candidate_ids"]:
        raise AssertionError("Address-prefix blocker failed")
    if address_target.entity_id in address_result["v3_candidate_ids"]:
        raise AssertionError("Address-prefix smoke pair was already a V3 seed")

    for cap in (75, 100):
        cap_index = V4TargetIndex("us", max_candidates=cap)
        for number in range(120):
            cap_index.add(Record(
                f"S2-cap-{number:03d}", "Cap Test Holdings", f"Street {number}", "US", "S2"
            ))
        cap_result = cap_index.candidates_with_sources(
            Record("S1-cap", "Cap Test Holdings", "Street Unknown", "US", "S1")
        )
        if len(cap_result["candidate_ids"]) != cap:
            raise AssertionError(f"Candidate cap {cap} was not applied exactly")
        if len(cap_result["candidate_ids"]) != len(set(cap_result["candidate_ids"])):
            raise AssertionError(f"Candidate identity deduplication failed at cap {cap}")

    # The same target identity proposed by multiple paths appears once in the output.
    duplicate_index = V4TargetIndex("us", max_candidates=75)
    duplicate_index.add(Record("S2-duplicate", "Repeated Store LLC", "Same Place", "US", "S2"))
    duplicate_index.add(Record("S2-duplicate", "Repeated Store LLC", "Same Place", "US", "S2"))
    duplicate_result = duplicate_index.candidates_with_sources(
        Record("S1-duplicate", "Repeated Store LLC", "Same Place", "US", "S1")
    )
    if duplicate_result["candidate_ids"].count("S2-duplicate") != 1:
        raise AssertionError("Duplicate target entity_id was emitted more than once")

    feature_index = address_index
    feature_idx = feature_index.v3_index.base.ids.index(address_target.entity_id)
    feature_vector = pair_features_v2(
        prepare_source1_v2(address_source), feature_index.feature_record(feature_idx)
    )
    if len(feature_vector) != 29 or not all(math.isfinite(float(v)) for v in feature_vector):
        raise AssertionError("The unchanged V3 29-feature pipeline rejected a V4 target")

    default_config = V4Config(Path(project_root))
    v3_artifacts = default_config.base_artifacts_dir / "v3"
    if default_config.artifacts_dir == v3_artifacts or v3_artifacts in default_config.artifacts_dir.parents:
        raise AssertionError("V4 artifact path overlaps the V3 artifact directory")
    if default_config.artifacts_dir == default_config.base_artifacts_dir:
        raise AssertionError("V4 artifact path is not isolated")

    return {
        "status": "PASS",
        "dataset_schema": schema,
        "v3_feature_count": len(feature_vector),
        "new_blockers": [
            "bounded single 3-character name-prefix rescue",
            "normalized legal-suffix-equivalent exact-name probes",
            "bounded 4-character address-token-prefix postings",
        ],
        "candidate_caps_checked": [75, 100],
        "candidate_identity_deduplication": "PASS",
        "v4_artifacts_dir": str(default_config.artifacts_dir),
        "v3_artifacts_dir_untouched": True,
        "full_dataset_read_or_training": False,
    }
