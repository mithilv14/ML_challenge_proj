import hashlib
import json
import random
from pathlib import Path

from .blocking import TargetIndex
from .config import PipelineConfig
from .data import iter_source, load_ground_truth
from .evaluate import ValidationAccumulator
from .features import FEATURE_NAMES, pair_features
from .model import LogisticRegressionSGD
from .normalize import normalize_address, normalize_country, normalize_name
from .predict import write_test_predictions


def _stable_hash(value):
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def _sample_records(source1_path, country, config):
    train, validation = [], []
    train_low, train_high = config.train_hash_range
    val_low, val_high = config.validation_hash_range
    for record in iter_source(source1_path, "S1", country):
        bucket = _stable_hash(record.entity_id) % config.hash_buckets
        if val_low <= bucket < val_high:
            validation.append((_stable_hash(record.entity_id), record))
        elif train_low <= bucket < train_high:
            train.append((_stable_hash(record.entity_id), record))
    train.sort(key=lambda item: item[0])
    validation.sort(key=lambda item: item[0])
    train = [record for _, record in train[:config.max_train_per_country]]
    validation = [record for _, record in validation[:config.max_validation_per_country]]
    return train, validation


def _candidate_feature_rows(source1, index, truth, rng, negative_cap, negative_ratio):
    candidates = index.candidates(source1)
    truth = set(truth)
    positives, negatives = [], []
    positive_ids = set()
    for idx in candidates:
        entity_id = index.ids[idx]
        if entity_id in truth:
            positive_ids.add(entity_id)
            positives.append((idx, entity_id))
        else:
            negatives.append(idx)
    n_negative = min(len(negatives), max(negative_cap, len(positives) * negative_ratio))
    sampled_negatives = rng.sample(negatives, n_negative) if n_negative else []
    n1, a1 = normalize_name(source1.business_name), normalize_address(source1.business_address)
    row1 = (n1, source1.entity_id, a1, source1.source)
    examples = []
    for idx, entity_id in positives:
        examples.append((pair_features(row1, index.record(idx), True), 1, entity_id))
    for idx in sampled_negatives:
        entity_id = index.ids[idx]
        examples.append((pair_features(row1, index.record(idx), True), 0, entity_id))
    return candidates, positive_ids, examples


def _fit_country(train_records, labels, index, config):
    model = LogisticRegressionSGD(len(FEATURE_NAMES))
    positives = 0
    sampled_negatives = 0
    for position, record in enumerate(train_records, start=1):
        truth = labels.get(record.entity_id, set())
        seed = _stable_hash(record.entity_id) ^ config.seed
        rng = random.Random(seed)
        _, _, examples = _candidate_feature_rows(
            record, index, truth, rng,
            config.negative_samples_per_entity,
            config.negative_to_positive_ratio,
        )
        rng.shuffle(examples)
        for features, label, _ in examples:
            model.update(features, label)
            if label:
                positives += 1
            else:
                sampled_negatives += 1
        if position % 5000 == 0:
            print(f"  trained on {position}/{len(train_records)} S1 rows", flush=True)
    return model, {"positive_pairs_seen": positives, "negative_pairs_sampled": sampled_negatives,
                   "model_updates": model.updates}


def _validate_country(validation_records, labels, index, model, thresholds):
    accumulator = ValidationAccumulator(thresholds)
    for position, record in enumerate(validation_records, start=1):
        candidates = index.candidates(record)
        n1, a1 = normalize_name(record.business_name), normalize_address(record.business_address)
        row1 = (n1, record.entity_id, a1, record.source)
        candidate_ids = [index.ids[idx] for idx in candidates]
        scores = [model.predict_proba(pair_features(row1, index.record(idx), True)) for idx in candidates]
        accumulator.add(labels.get(record.entity_id, set()), candidate_ids, scores)
        if position % 1000 == 0:
            print(f"  validated {position}/{len(validation_records)} S1 rows", flush=True)
    return accumulator.report(target_count=len(index))


def _average_models(models, weights):
    if not models:
        return LogisticRegressionSGD(len(FEATURE_NAMES))
    total = sum(weights) or len(weights)
    result = LogisticRegressionSGD(len(FEATURE_NAMES))
    result.weights = [
        sum(model.weights[i] * weight for model, weight in zip(models, weights)) / total
        for i in range(len(FEATURE_NAMES))
    ]
    result.intercept = sum(model.intercept * weight for model, weight in zip(models, weights)) / total
    result.updates = sum(model.updates for model in models)
    return result


def train_pipeline(config=None):
    config = config or PipelineConfig()
    train_dir = config.data_dir / "train"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"
    countries = set()
    for record in iter_source(s1_path, "S1"):
        countries.add(normalize_country(record.country))
    countries.discard("")

    models = {}
    threshold_by_country = {}
    country_reports = {}
    model_weights = []
    country_models = []
    for country in sorted(countries):
        print(f"Preparing training sample for country={country}", flush=True)
        train_records, validation_records = _sample_records(s1_path, country, config)
        wanted_ids = {record.entity_id for record in train_records + validation_records}
        labels = load_ground_truth(gt_path, wanted_ids)
        missing_labels = wanted_ids - set(labels)
        if missing_labels:
            raise ValueError(f"Ground truth is missing {len(missing_labels)} sampled S1 rows")

        print(f"Building target index for country={country}", flush=True)
        index = TargetIndex(
            country,
            max_token_posting=config.max_token_posting,
            max_candidates=config.max_candidates_per_entity,
        ).extend_from_files(s2_path, s3_path)
        print(f"Indexed {len(index)} S2/S3 records for country={country}", flush=True)
        model, train_stats = _fit_country(train_records, labels, index, config)
        report = _validate_country(validation_records, labels, index, model, config.thresholds)
        report["train_entities"] = len(train_records)
        report["train_stats"] = train_stats
        report["target_entities"] = len(index)
        report["country"] = country
        models[country] = model
        threshold_by_country[country] = float(report["best_threshold"])
        country_reports[country] = report
        country_models.append(model)
        model_weights.append(max(1, train_stats["model_updates"]))
        print(f"Validated {country}: n_train={len(train_records)} n_validation={len(validation_records)} "
              f"blocking_recall={report['blocking_recall']:.4f} macro_f0.5={report['macro_f0_5']:.4f} "
              f"threshold={report['best_threshold']}", flush=True)

    fallback = _average_models(country_models, model_weights)
    fallback_threshold = sum(threshold_by_country.values()) / len(threshold_by_country) if threshold_by_country else 0.75
    all_validation = sum(report["entities"] for report in country_reports.values())
    weighted_f = sum(report["macro_f0_5"] * report["entities"] for report in country_reports.values())
    total_tp = sum(report["true_positive"] for report in country_reports.values())
    total_predicted = sum(report["predicted_pairs"] for report in country_reports.values())
    total_true = sum(report["true_matches"] for report in country_reports.values())
    total_singletons = sum(report["singleton_entities"] for report in country_reports.values())
    singleton_correct = sum(report["singleton_correct"] for report in country_reports.values())
    blocked_true = sum(report["blocked_true_matches"] for report in country_reports.values())
    total_candidates = sum(report["candidate_pairs"] for report in country_reports.values())
    total_possible = sum(report["entities"] * report["target_entities"] for report in country_reports.values())
    overall = {
        "validation_entities": all_validation,
        "macro_f0_5_country_tuned": weighted_f / all_validation if all_validation else 0.0,
        "precision_micro_country_tuned": total_tp / total_predicted if total_predicted else 0.0,
        "recall_micro_country_tuned": total_tp / total_true if total_true else 0.0,
        "blocking_recall": blocked_true / total_true if total_true else 0.0,
        "singleton_accuracy": singleton_correct / total_singletons if total_singletons else 0.0,
        "candidate_pairs": total_candidates,
        "average_candidates_per_entity": total_candidates / all_validation if all_validation else 0.0,
        "maximum_candidates_per_entity": max((report["maximum_candidates_per_entity"] for report in country_reports.values()), default=0),
        "candidate_reduction_ratio": 1.0 - total_candidates / total_possible if total_possible else 0.0,
        "country_reports": country_reports,
        "threshold_by_country": threshold_by_country,
        "fallback_threshold_for_unseen_country": fallback_threshold,
        "feature_names": list(FEATURE_NAMES),
        "sampling": {
            "hash_buckets": config.hash_buckets,
            "train_hash_range": list(config.train_hash_range),
            "validation_hash_range": list(config.validation_hash_range),
            "max_train_per_country": config.max_train_per_country,
            "max_validation_per_country": config.max_validation_per_country,
            "max_candidates_per_entity": config.max_candidates_per_entity,
        },
        "baseline": "country-partitioned blocking + pairwise logistic regression (standard-library SGD)",
    }
    artifact = {
        "models": {country: model.to_dict() for country, model in models.items()},
        "fallback_model": fallback.to_dict(),
        "threshold_by_country": threshold_by_country,
        "fallback_threshold": fallback_threshold,
        "feature_names": list(FEATURE_NAMES),
        "seed": config.seed,
    }
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (config.artifacts_dir / "baseline_model.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.artifacts_dir / "validation_report.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact, overall


def predict_pipeline(config=None, artifact_path=None, output_dir=None, threshold_override=None):
    config = config or PipelineConfig()
    artifact_path = Path(artifact_path or config.artifacts_dir / "baseline_model.json")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    return write_test_predictions(config, artifact, output_dir, threshold_override)
