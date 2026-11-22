import array
import hashlib
import json
import math
import random
from pathlib import Path
from time import perf_counter

from ..config import PipelineConfig
from ..data import iter_source, load_ground_truth
from ..normalize import normalize_country
from ..pipeline import _sample_records
from ..evaluate import ValidationAccumulator
from ..v2.blocking import V2TargetIndex
from ..v2.evaluate import evaluate_cached_rows, tuned_overall
from ..v2.features import V2_FEATURE_NAMES, pair_features_v2, prepare_source1_v2
from .config import POSITIVE_SAMPLE_WEIGHT, THRESHOLDS, V2_REFERENCE, XGBOOST_PARAMS


def _stable_hash(value):
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def _load_xgboost():
    """Load the requested implementation lazily; never substitute another model."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required to train V3. Install the 'xgboost' package in this "
            "environment (for example, in Kaggle), then rerun. V3 has no fallback scorer."
        ) from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("V3 requires NumPy, which is normally installed with XGBoost.") from exc
    return XGBClassifier, np


def _append_features(flat_features, labels, weights, feature_vector, label):
    if len(feature_vector) != len(V2_FEATURE_NAMES):
        raise ValueError(
            f"V2 feature count changed: got {len(feature_vector)}, expected {len(V2_FEATURE_NAMES)}"
        )
    if not all(math.isfinite(float(value)) for value in feature_vector):
        raise ValueError("Non-finite pair feature encountered")
    flat_features.extend(float(value) for value in feature_vector)
    labels.append(int(label))
    weights.append(POSITIVE_SAMPLE_WEIGHT if label else 1.0)


def _fit_country(train_records, labels_by_s1, index, config, XGBClassifier, np):
    if array.array("f").itemsize != 4:
        raise RuntimeError("This runtime does not provide 32-bit C floats for compact V3 matrices")
    if array.array("i").itemsize != 4:
        raise RuntimeError("This runtime does not provide 32-bit C integers for compact V3 labels")
    flat_features = array.array("f")
    label_values = array.array("i")
    sample_weights = array.array("f")
    candidate_pairs = positive_pairs = negative_pairs = 0

    for position, source1 in enumerate(train_records, start=1):
        truth = labels_by_s1.get(source1.entity_id, set())
        candidates, _, _ = index.candidates_with_comparison(source1)
        candidate_pairs += len(candidates)
        positives, negatives = [], []
        for idx in candidates:
            if index.base.ids[idx] in truth:
                positives.append(idx)
            else:
                negatives.append(idx)

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
            feature_vector = pair_features_v2(prepared, index.feature_record(idx))
            _append_features(flat_features, label_values, sample_weights, feature_vector, label)
            positive_pairs += label
            negative_pairs += 1 - label

        if position % 1000 == 0:
            print(
                f"  V3 prepared {position:,}/{len(train_records):,} {index.country} S1 rows",
                flush=True,
            )

    n_features = len(V2_FEATURE_NAMES)
    row_count = len(label_values)
    if row_count == 0 or not positive_pairs or not negative_pairs:
        raise ValueError(
            f"Insufficient sampled training pairs for {index.country}: "
            f"positive={positive_pairs}, negative={negative_pairs}"
        )
    X = np.frombuffer(flat_features, dtype=np.float32).reshape(row_count, n_features)
    y = np.frombuffer(label_values, dtype=np.int32)
    weights = np.frombuffer(sample_weights, dtype=np.float32)
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model, {
        "training_candidate_pairs_before_sampling": candidate_pairs,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "training_examples": row_count,
        "positive_sample_weight": POSITIVE_SAMPLE_WEIGHT,
    }


def _load_or_build_validation_rows(country, validation_records, index, config, cache, truth_path):
    """Prefer the exact V2 candidate cache; regenerate only if it is unavailable."""
    if cache is not None:
        cached_country = cache.get("countries", {}).get(country)
        if cached_country is None:
            raise ValueError(f"V2 validation cache has no country entry for {country!r}")
        rows = cached_country["rows"]
        expected_ids = [record.entity_id for record in validation_records]
        cached_ids = [row.get("s1_id") for row in rows]
        if cached_ids != expected_ids:
            raise ValueError(
                f"V2 cache validation S1 rows do not match the configured V2 split for {country}; "
                "refusing to compare on a different sample"
            )
        if int(cached_country.get("target_entities", -1)) != len(index):
            raise ValueError(f"V2 cache target count differs from the rebuilt {country} index")
        for row in rows:
            candidate_ids = row.get("v2_candidate_ids", [])
            if len(candidate_ids) > config.max_candidates_per_entity:
                raise ValueError(f"Cached V2 row {row.get('s1_id')} exceeds the candidate cap")
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"Cached V2 row {row.get('s1_id')} has duplicate candidates")
            if len(candidate_ids) != len(row.get("v2_scores", [])):
                raise ValueError(f"Cached V2 scores are misaligned for {row.get('s1_id')}")
        return rows, True

    wanted = {record.entity_id for record in validation_records}
    truth = load_ground_truth(truth_path, wanted)
    missing = wanted - set(truth)
    if missing:
        raise ValueError(f"Ground truth is missing {len(missing)} validation rows for {country}")
    rows = []
    for record in validation_records:
        candidate_indices, _, _ = index.candidates_with_comparison(record)
        rows.append({
            "s1_id": record.entity_id,
            "truth_ids": sorted(truth.get(record.entity_id, set())),
            "v2_candidate_ids": [index.base.ids[idx] for idx in candidate_indices],
        })
    return rows, False


def _score_validation(validation_records, rows, index, model, np, country):
    requested_ids = {
        candidate_id
        for row in rows
        for candidate_id in row["v2_candidate_ids"]
    }
    candidate_index = {
        entity_id: idx
        for idx, entity_id in enumerate(index.base.ids)
        if entity_id in requested_ids
    }
    missing = requested_ids - set(candidate_index)
    if missing:
        raise ValueError(
            f"{len(missing)} cached V2 candidates are absent from the rebuilt {country} target index"
        )

    flat_features = array.array("f")
    row_sizes = []
    for position, (record, row) in enumerate(zip(validation_records, rows), start=1):
        if row["s1_id"] != record.entity_id:
            raise ValueError("Validation rows lost alignment with the V2 cache")
        prepared = prepare_source1_v2(record)
        candidate_ids = row["v2_candidate_ids"]
        row_sizes.append(len(candidate_ids))
        for candidate_id in candidate_ids:
            vector = pair_features_v2(
                prepared,
                index.feature_record(candidate_index[candidate_id]),
            )
            if len(vector) != len(V2_FEATURE_NAMES) or not all(
                math.isfinite(float(value)) for value in vector
            ):
                raise ValueError("Invalid V2 feature vector while scoring validation candidates")
            flat_features.extend(float(value) for value in vector)
        if position % 500 == 0:
            print(
                f"  V3 prepared validation features {position:,}/{len(validation_records):,} "
                f"{country} S1 rows",
                flush=True,
            )

    total_pairs = sum(row_sizes)
    if total_pairs:
        X = np.frombuffer(flat_features, dtype=np.float32).reshape(total_pairs, len(V2_FEATURE_NAMES))
        probabilities = model.predict_proba(X)[:, 1].astype(float).tolist()
    else:
        probabilities = []

    scores_by_row = []
    offset = 0
    for size in row_sizes:
        scores_by_row.append(probabilities[offset:offset + size])
        offset += size
    return scores_by_row


def train_v3_pipeline(config=None):
    """Train the isolated V3 XGBoost scorer on the unchanged V2 pipeline inputs."""
    XGBClassifier, np = _load_xgboost()  # Fail before opening data or building indexes.
    started = perf_counter()
    config = config or PipelineConfig()
    config = PipelineConfig(
        project_root=config.project_root,
        max_train_per_country=config.max_train_per_country,
        max_validation_per_country=config.max_validation_per_country,
        train_hash_range=config.train_hash_range,
        validation_hash_range=config.validation_hash_range,
        hash_buckets=config.hash_buckets,
        negative_samples_per_entity=config.negative_samples_per_entity,
        negative_to_positive_ratio=config.negative_to_positive_ratio,
        seed=config.seed,
        max_token_posting=config.max_token_posting,
        max_candidates_per_entity=config.max_candidates_per_entity,
        thresholds=THRESHOLDS,
    )
    train_dir = config.data_dir / "train"
    source1_path = train_dir / "train_source1.tsv"
    source2_path = train_dir / "train_source2.tsv"
    source3_path = train_dir / "train_source3.tsv"
    truth_path = train_dir / "train_ground_truth.tsv"
    artifact_dir = config.artifacts_dir
    v2_cache_path = artifact_dir / "v2_validation_cache.json"
    v2_report_path = artifact_dir / "v2_validation_report.json"
    if not v2_report_path.is_file():
        raise FileNotFoundError(f"Missing V2 report needed for comparison: {v2_report_path}")
    v2_saved_report = json.loads(v2_report_path.read_text(encoding="utf-8"))
    v2_cache = (
        json.loads(v2_cache_path.read_text(encoding="utf-8"))
        if v2_cache_path.is_file()
        else None
    )
    if v2_cache is not None:
        saved_sampling = v2_saved_report.get("sampling", {})
        expected_sampling = {
            "hash_buckets": config.hash_buckets,
            "train_hash_range": list(config.train_hash_range),
            "validation_hash_range": list(config.validation_hash_range),
            "max_train_per_country": config.max_train_per_country,
            "max_validation_per_country": config.max_validation_per_country,
            "max_candidates_per_entity": config.max_candidates_per_entity,
        }
        mismatches = {
            key: (saved_sampling.get(key), expected)
            for key, expected in expected_sampling.items()
            if saved_sampling.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "V3 settings differ from the saved V2 sampling/candidate configuration; "
                f"refusing a non-comparable run: {mismatches}"
            )

    countries = sorted({
        normalize_country(record.country)
        for record in iter_source(source1_path, "S1")
        if normalize_country(record.country)
    })
    country_reports = {}
    country_score_cache = {}
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for country in countries:
        print(f"Preparing V3 training sample for country={country}", flush=True)
        train_records, validation_records = _sample_records(source1_path, country, config)
        train_ids = {record.entity_id for record in train_records}
        validation_ids = {record.entity_id for record in validation_records}
        if train_ids & validation_ids:
            raise ValueError(f"Train/validation entity split overlaps for {country}")
        labels = load_ground_truth(truth_path, train_ids)
        missing_labels = train_ids - set(labels)
        if missing_labels:
            raise ValueError(f"Ground truth is missing {len(missing_labels)} training rows for {country}")

        print(f"Building the unchanged V2 target index for country={country}", flush=True)
        index = V2TargetIndex(
            country,
            max_token_posting=config.max_token_posting,
            max_candidates=config.max_candidates_per_entity,
        ).extend_from_files(source2_path, source3_path)
        print(f"Indexed {len(index):,} S2/S3 records for country={country}", flush=True)

        model, training_stats = _fit_country(
            train_records, labels, index, config, XGBClassifier, np
        )
        model_path = artifact_dir / f"v3_xgboost_{country}.json"
        model.save_model(str(model_path))

        validation_rows, reused_cache = _load_or_build_validation_rows(
            country, validation_records, index, config, v2_cache, truth_path
        )
        if reused_cache:
            # Ground truth is read only after fitting, and is used only for evaluation.
            v2_metrics = evaluate_cached_rows(
                validation_rows, THRESHOLDS, len(index)
            )["v2"]
        else:
            # Candidate regeneration path has no cached V2 scores; evaluate V3 only.
            v2_metrics = None
        score_rows = _score_validation(
            validation_records, validation_rows, index, model, np, country
        )
        accumulator = ValidationAccumulator(THRESHOLDS)
        v3_rows = []
        for row, scores in zip(validation_rows, score_rows):
            candidate_ids = row["v2_candidate_ids"]
            if len(candidate_ids) != len(scores):
                raise ValueError(f"V3 score count differs from candidate count for {row['s1_id']}")
            truth_ids = row["truth_ids"]
            accumulator.add(truth_ids, candidate_ids, scores)
            v3_rows.append({
                "s1_id": row["s1_id"],
                "truth_ids": truth_ids,
                "candidate_ids": candidate_ids,
                "v3_scores": scores,
            })
        v3_metrics = accumulator.report(target_count=len(index))

        country_cache = {
            "target_entities": len(index),
            "validation_candidate_source": "v2_validation_cache" if reused_cache else "regenerated_v2_blocker",
            "rows": v3_rows,
        }
        country_score_cache[country] = country_cache
        v2_country = (
            v2_metrics
            if v2_metrics is not None
            else v2_saved_report.get("country_reports", {}).get(country, {}).get("v2")
        )
        if v2_country is None:
            v2_country = V2_REFERENCE["countries"].get(country)
        country_reports[country] = {
            "country": country,
            "train_entities": len(train_records),
            "training": training_stats,
            "target_entities": len(index),
            "validation_candidate_source": country_cache["validation_candidate_source"],
            "v2": v2_country,
            "v3": v3_metrics,
            "model_path": str(model_path),
        }
        print(
            f"V3 validated {country}: macro_f0.5={v3_metrics['macro_f0_5']:.4f} "
            f"precision={v3_metrics['precision_micro']:.4f} "
            f"recall={v3_metrics['recall_micro']:.4f} "
            f"threshold={v3_metrics['best_threshold']} "
            f"candidates={v3_metrics['candidate_pairs']:,}",
            flush=True,
        )

    v3_overall = tuned_overall({
        country: detail["v3"] for country, detail in country_reports.items()
    })
    if all(detail["v2"] is not None for detail in country_reports.values()):
        v2_overall = tuned_overall({
            country: detail["v2"] for country, detail in country_reports.items()
        })
    else:
        v2_overall = v2_saved_report.get("v2", V2_REFERENCE["overall"])
    threshold_by_country = {
        country: detail["v3"]["best_threshold"]
        for country, detail in country_reports.items()
    }
    runtime_seconds = round(perf_counter() - started, 2)
    report = {
        "version": "V3 XGBoost scorer with unchanged V2 candidates/features",
        "runtime_seconds": runtime_seconds,
        "runtime_minutes": round(runtime_seconds / 60.0, 2),
        "xgboost_params": XGBOOST_PARAMS,
        "positive_sample_weight": POSITIVE_SAMPLE_WEIGHT,
        "feature_names": list(V2_FEATURE_NAMES),
        "thresholds": list(THRESHOLDS),
        "threshold_selection": "Per-country maximum validation macro F0.5; ties select the higher threshold, matching the shared evaluator.",
        "sampling": {
            "hash_buckets": config.hash_buckets,
            "train_hash_range": list(config.train_hash_range),
            "validation_hash_range": list(config.validation_hash_range),
            "max_train_per_country": config.max_train_per_country,
            "max_validation_per_country": config.max_validation_per_country,
            "max_candidates_per_entity": config.max_candidates_per_entity,
            "negative_samples_per_entity": config.negative_samples_per_entity,
            "negative_to_positive_ratio": config.negative_to_positive_ratio,
            "seed": config.seed,
        },
        "v2_reference": v2_overall,
        "v2_reference_saved": v2_saved_report.get("v2", V2_REFERENCE["overall"]),
        "v3": v3_overall,
        "delta_v3_minus_v2": {
            "macro_f0_5_country_tuned": v3_overall["macro_f0_5_country_tuned"] - v2_overall["macro_f0_5_country_tuned"],
            "precision_micro_country_tuned": v3_overall["precision_micro_country_tuned"] - v2_overall["precision_micro_country_tuned"],
            "recall_micro_country_tuned": v3_overall["recall_micro_country_tuned"] - v2_overall["recall_micro_country_tuned"],
        },
        "threshold_by_country": threshold_by_country,
        "country_reports": country_reports,
        "models": {
            country: detail["model_path"] for country, detail in country_reports.items()
        },
        "notes": [
            "V2 normalization, candidate generation, 50-candidate cap, pair feature implementation, split, and deterministic negative sampling were reused unchanged.",
            "The existing V2 validation candidate cache is preferred and checked against the deterministic validation S1 sample.",
            "XGBoost was used as the only V3 scorer; no model fallback is provided.",
        ],
    }
    (artifact_dir / "v3_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    score_cache = {
        "version": "V3 validation score cache",
        "thresholds": list(THRESHOLDS),
        "countries": country_score_cache,
    }
    (artifact_dir / "v3_validation_scores.json").write_text(
        json.dumps(score_cache, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    return report


def static_check(project_root):
    """Validate V3 wiring and saved V2 inputs without reading datasets or training."""
    from ..data import Record
    from ..v2.features import pair_features_v2, prepare_source1_v2
    from ..v2.blocking import V2TargetIndex

    package_dir = Path(__file__).resolve().parents[3]
    artifacts = Path(project_root) / "code" / "business_entity_resolution" / "artifacts"
    v2_report_path = artifacts / "v2_validation_report.json"
    v2_cache_path = artifacts / "v2_validation_cache.json"
    if not v2_report_path.is_file() or not v2_cache_path.is_file():
        raise FileNotFoundError("Static check requires the saved V2 report and validation cache")
    saved = json.loads(v2_report_path.read_text(encoding="utf-8"))
    default_config = PipelineConfig(project_root=Path(project_root))
    if saved.get("feature_names") != list(V2_FEATURE_NAMES):
        raise ValueError("V3 feature schema does not match the saved V2 validation report")
    saved_sampling = saved.get("sampling", {})
    expected_sampling = {
        "hash_buckets": default_config.hash_buckets,
        "train_hash_range": list(default_config.train_hash_range),
        "validation_hash_range": list(default_config.validation_hash_range),
        "max_train_per_country": default_config.max_train_per_country,
        "max_validation_per_country": default_config.max_validation_per_country,
        "max_candidates_per_entity": default_config.max_candidates_per_entity,
    }
    if any(saved_sampling.get(key) != value for key, value in expected_sampling.items()):
        raise ValueError("Default V3 sampling differs from the recorded V2 comparison configuration")
    if len(V2_FEATURE_NAMES) != 29:
        raise ValueError(f"Expected the preserved V2 29-feature schema, found {len(V2_FEATURE_NAMES)}")
    if tuple(THRESHOLDS) != (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        raise ValueError("V3 threshold sweep differs from the requested 0.50–0.95 range")
    index = V2TargetIndex("us", max_candidates=50)
    target = Record("S2-static-1", "Northstar LLC", "10 Main Street", "US", "S2")
    source = Record("S1-static-1", "Northstar, L.L.C.", "10 Main St.", "US", "S1")
    index.add(target)
    vector = pair_features_v2(prepare_source1_v2(source), index.feature_record(0))
    if len(vector) != 29 or not all(math.isfinite(float(value)) for value in vector):
        raise ValueError("V3 shared feature smoke check failed")
    try:
        import importlib.util
        xgboost_available = importlib.util.find_spec("xgboost") is not None
    except (ImportError, ValueError):
        xgboost_available = False
    return {
        "status": "PASS",
        "project_root": str(Path(project_root).resolve()),
        "package_root": str(package_dir),
        "features": len(V2_FEATURE_NAMES),
        "thresholds": list(THRESHOLDS),
        "v2_report": str(v2_report_path),
        "v2_validation_cache": str(v2_cache_path),
        "xgboost_available_for_training": xgboost_available,
        "training_run_performed": False,
    }
