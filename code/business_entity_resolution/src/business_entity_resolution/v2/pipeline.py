import hashlib
import json
import random
from pathlib import Path
from time import perf_counter

from ..config import PipelineConfig
from ..data import iter_source, load_ground_truth
from ..features import pair_features as baseline_pair_features
from ..model import LogisticRegressionSGD
from ..normalize import normalize_address, normalize_country, normalize_name
from ..pipeline import _average_models, _sample_records
from .blocking import V2TargetIndex
from .evaluate import evaluate_cached_rows, tuned_overall
from .features import V2_FEATURE_NAMES, pair_features_v2, prepare_source1_v2


def _stable_hash(value):
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def _fit_country(train_records, labels, index, config):
    model = LogisticRegressionSGD(len(V2_FEATURE_NAMES))
    positive_pairs = 0
    sampled_negatives = 0
    candidate_total = 0
    for position, source1 in enumerate(train_records, start=1):
        truth = labels.get(source1.entity_id, set())
        candidate_indices, _, _ = index.candidates_with_comparison(source1)
        candidate_total += len(candidate_indices)
        positives, negatives = [], []
        for idx in candidate_indices:
            if index.base.ids[idx] in truth:
                positives.append(idx)
            else:
                negatives.append(idx)
        n_negative = min(
            len(negatives),
            max(config.negative_samples_per_entity,
                len(positives) * config.negative_to_positive_ratio),
        )
        seed = _stable_hash(source1.entity_id) ^ config.seed
        rng = random.Random(seed)
        selected = [(idx, 1) for idx in positives]
        selected.extend((idx, 0) for idx in (rng.sample(negatives, n_negative) if n_negative else []))
        rng.shuffle(selected)
        prepared = prepare_source1_v2(source1)
        for idx, label in selected:
            model.update(pair_features_v2(prepared, index.feature_record(idx)), label)
            positive_pairs += int(bool(label))
            sampled_negatives += int(not label)
        if position % 5000 == 0:
            print(f"  V2 trained on {position:,}/{len(train_records):,} {index.country} S1 rows", flush=True)
    return model, {
        "positive_pairs_seen": positive_pairs,
        "negative_pairs_sampled": sampled_negatives,
        "model_updates": model.updates,
        "candidate_pairs_before_negative_sampling": candidate_total,
    }


def _score_validation(validation_records, labels, index, v2_model, baseline_model):
    rows = []
    stats = {
        "baseline_candidates": 0,
        "v2_candidates": 0,
        "v2_candidates_before_cap": 0,
        "additional_candidates_before_cap": 0,
        "baseline_capped_entities": 0,
        "v2_capped_entities": 0,
    }
    for position, source1 in enumerate(validation_records, start=1):
        v2_indices, baseline_indices, block_stats = index.candidates_with_comparison(source1)
        prepared = prepare_source1_v2(source1)
        truth = sorted(labels.get(source1.entity_id, set()))
        baseline_ids = [index.base.ids[idx] for idx in baseline_indices]
        v2_ids = [index.base.ids[idx] for idx in v2_indices]
        baseline_scores = [
            baseline_model.predict_proba(
                baseline_pair_features(prepared["base"], index.record(idx), True)
            )
            for idx in baseline_indices
        ]
        v2_scores = [
            v2_model.predict_proba(pair_features_v2(prepared, index.feature_record(idx)))
            for idx in v2_indices
        ]
        rows.append({
            "s1_id": source1.entity_id,
            "truth_ids": truth,
            "baseline_candidate_ids": baseline_ids,
            "baseline_scores": baseline_scores,
            "v2_candidate_ids": v2_ids,
            "v2_scores": v2_scores,
        })
        stats["baseline_candidates"] += len(baseline_ids)
        stats["v2_candidates"] += len(v2_ids)
        stats["v2_candidates_before_cap"] += block_stats["v2_pre_cap_candidates"]
        stats["additional_candidates_before_cap"] += block_stats["additional_pre_cap_candidates"]
        stats["baseline_capped_entities"] += int(block_stats["baseline_capped"])
        stats["v2_capped_entities"] += int(block_stats["v2_capped"])
        if position % 1000 == 0:
            print(f"  V2 scored {position:,}/{len(validation_records):,} {index.country} validation S1 rows", flush=True)
    return rows, stats


def _combine_country_reports(reports, model_name):
    return tuned_overall({country: value[model_name] for country, value in reports.items()})


def train_v2_pipeline(config=None):
    """Train and evaluate isolated V2; baseline files are read-only inputs."""
    started = perf_counter()
    config = config or PipelineConfig()
    train_dir = config.data_dir / "train"
    source1_path = train_dir / "train_source1.tsv"
    source2_path = train_dir / "train_source2.tsv"
    source3_path = train_dir / "train_source3.tsv"
    truth_path = train_dir / "train_ground_truth.tsv"
    artifact_dir = config.artifacts_dir
    baseline_model_path = artifact_dir / "baseline_model.json"
    baseline_report_path = artifact_dir / "validation_report.json"
    if not baseline_model_path.is_file() or not baseline_report_path.is_file():
        raise FileNotFoundError("Expected preserved baseline model and validation report under artifacts/")

    baseline_artifact = json.loads(baseline_model_path.read_text(encoding="utf-8"))
    saved_baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    baseline_models = {
        country: LogisticRegressionSGD.from_dict(payload)
        for country, payload in baseline_artifact["models"].items()
    }
    baseline_fallback = LogisticRegressionSGD.from_dict(baseline_artifact["fallback_model"])

    countries = sorted({
        normalize_country(record.country)
        for record in iter_source(source1_path, "S1")
        if normalize_country(record.country)
    })
    models = {}
    country_details = {}
    country_cache = {}
    model_weights = []
    model_list = []

    for country in countries:
        print(f"Preparing V2 sample for country={country}", flush=True)
        train_records, validation_records = _sample_records(source1_path, country, config)
        wanted_ids = {record.entity_id for record in train_records + validation_records}
        labels = load_ground_truth(truth_path, wanted_ids)
        missing_labels = wanted_ids - set(labels)
        if missing_labels:
            raise ValueError(f"Ground truth is missing {len(missing_labels)} sampled S1 rows")

        print(f"Building shared V2/baseline comparison index for country={country}", flush=True)
        index = V2TargetIndex(
            country,
            max_token_posting=config.max_token_posting,
            max_candidates=config.max_candidates_per_entity,
        ).extend_from_files(source2_path, source3_path)
        print(f"Indexed {len(index):,} S2/S3 records for country={country}", flush=True)

        model, train_stats = _fit_country(train_records, labels, index, config)
        baseline_model = baseline_models.get(country, baseline_fallback)
        rows, candidate_stats = _score_validation(
            validation_records, labels, index, model, baseline_model
        )
        evaluations = evaluate_cached_rows(rows, config.thresholds, len(index))
        country_cache[country] = {"target_entities": len(index), "rows": rows}
        detail = {
            "baseline": evaluations["baseline"],
            "v2": evaluations["v2"],
            "train_entities": len(train_records),
            "train_stats": train_stats,
            "target_entities": len(index),
            "candidate_generation": candidate_stats,
            "country": country,
        }
        country_details[country] = detail
        models[country] = model
        model_list.append(model)
        model_weights.append(max(1, train_stats["model_updates"]))
        print(
            f"V2 validated {country}: blocking_recall={detail['v2']['blocking_recall']:.4f} "
            f"macro_f0.5={detail['v2']['macro_f0_5']:.4f} "
            f"threshold={detail['v2']['best_threshold']} "
            f"candidates={detail['v2']['candidate_pairs']:,}",
            flush=True,
        )

    baseline_overall = _combine_country_reports(country_details, "baseline")
    v2_overall = _combine_country_reports(country_details, "v2")
    baseline_threshold_by_country = {
        country: report["baseline"]["best_threshold"]
        for country, report in country_details.items()
    }
    v2_threshold_by_country = {
        country: report["v2"]["best_threshold"]
        for country, report in country_details.items()
    }
    fallback = _average_models(model_list, model_weights)
    fallback_threshold = (
        sum(v2_threshold_by_country.values()) / len(v2_threshold_by_country)
        if v2_threshold_by_country else 0.75
    )
    baseline_reference = {
        "macro_f0_5": saved_baseline_report["macro_f0_5_country_tuned"],
        "blocking_recall": saved_baseline_report["blocking_recall"],
        "precision_micro": saved_baseline_report["precision_micro_country_tuned"],
        "recall_micro": saved_baseline_report["recall_micro_country_tuned"],
        "candidate_pairs": saved_baseline_report["candidate_pairs"],
    }
    report = {
        "version": "V2",
        "runtime_seconds": None,
        "validation_entities": v2_overall["validation_entities"],
        "sampling": {
            "hash_buckets": config.hash_buckets,
            "train_hash_range": list(config.train_hash_range),
            "validation_hash_range": list(config.validation_hash_range),
            "max_train_per_country": config.max_train_per_country,
            "max_validation_per_country": config.max_validation_per_country,
            "max_candidates_per_entity": config.max_candidates_per_entity,
        },
        "baseline_saved_report": baseline_reference,
        "baseline_recomputed_same_validation": baseline_overall,
        "v2": v2_overall,
        "delta_vs_recomputed_baseline": {
            "macro_f0_5": v2_overall["macro_f0_5_country_tuned"] - baseline_overall["macro_f0_5_country_tuned"],
            "blocking_recall": v2_overall["blocking_recall"] - baseline_overall["blocking_recall"],
            "precision_micro": v2_overall["precision_micro_country_tuned"] - baseline_overall["precision_micro_country_tuned"],
            "recall_micro": v2_overall["recall_micro_country_tuned"] - baseline_overall["recall_micro_country_tuned"],
        },
        "baseline_threshold_by_country": baseline_threshold_by_country,
        "v2_threshold_by_country": v2_threshold_by_country,
        "feature_names": list(V2_FEATURE_NAMES),
        "country_reports": country_details,
        "notes": [
            "Baseline source files and artifacts were read-only during this run.",
            "Threshold metrics were swept from cached validation candidate scores; candidate generation and pair scoring were each performed once.",
            "Phone, email and postal signals are extracted conservatively from name/address text because the source schema has no dedicated contact columns.",
        ],
    }
    report["runtime_seconds"] = round(perf_counter() - started, 2)
    report["runtime_minutes"] = round(report["runtime_seconds"] / 60.0, 2)
    model_artifact = {
        "version": "V2",
        "models": {country: model.to_dict() for country, model in models.items()},
        "fallback_model": fallback.to_dict(),
        "threshold_by_country": v2_threshold_by_country,
        "fallback_threshold": fallback_threshold,
        "feature_names": list(V2_FEATURE_NAMES),
        "max_candidates_per_entity": config.max_candidates_per_entity,
        "normalization": "baseline cleanup plus terminal legal-form canonicalization, direction expansion, and conservative embedded contact/postal extraction",
    }
    cache_artifact = {
        "version": "V2 validation score cache",
        "thresholds": list(config.thresholds),
        "countries": country_cache,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "v2_model.json").write_text(
        json.dumps(model_artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_dir / "v2_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_dir / "v2_validation_cache.json").write_text(
        json.dumps(cache_artifact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    return model_artifact, report
