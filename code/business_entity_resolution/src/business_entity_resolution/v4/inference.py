"""Full test-set inference for the trained V4 XGBoost experiment."""

import array
import csv
import gc
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from ..data import iter_source
from ..normalize import normalize_country
from ..v2.features import V2_FEATURE_NAMES, pair_features_v2, prepare_source1_v2
from .blocking import V4TargetIndex
from .config import V4Config, XGBOOST_PARAMS


def _median_from_histogram(histogram, count):
    if not count:
        return 0.0
    ranks = ((count - 1) // 2, count // 2)
    values = []
    cumulative = 0
    rank_index = 0
    for value, frequency in enumerate(histogram):
        cumulative += frequency
        while rank_index < len(ranks) and ranks[rank_index] < cumulative:
            values.append(value)
            rank_index += 1
    return sum(values) / len(values)


def _load_models(artifact_dir):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "V4 inference requires XGBoost. Install code/business_entity_resolution/requirements.txt."
        ) from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("V4 inference requires NumPy.") from exc

    report_path = Path(artifact_dir) / "v4_validation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing V4 threshold report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model_reports = report.get("country_reports", {})
    models, thresholds = {}, {}
    for raw_country, details in sorted(model_reports.items()):
        country = normalize_country(raw_country)
        threshold = details.get("v4", {}).get("best_threshold")
        if threshold is None:
            raise ValueError(f"V4 validation report has no selected threshold for {raw_country!r}")
        model_path = Path(artifact_dir) / f"v4_xgboost_{country}.json"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing trained V4 model for {raw_country!r}: {model_path}")
        model = XGBClassifier(**XGBOOST_PARAMS)
        model.load_model(str(model_path))
        models[country] = model
        thresholds[country] = float(threshold)

    if not models:
        raise ValueError(f"No trained V4 country models found in {artifact_dir}")
    fallback_threshold = sum(thresholds.values()) / len(thresholds)
    return models, thresholds, fallback_threshold, np, report


def _country_summary(counts):
    rows = counts["source1_rows"]
    return {
        "source1_rows": rows,
        "target_records_indexed": counts["target_records_indexed"],
        "candidate_pairs": counts["candidate_pairs"],
        "average_candidates_per_source1": counts["candidate_pairs"] / rows if rows else 0.0,
        "median_candidates_per_source1": _median_from_histogram(counts["candidate_histogram"], rows),
        "maximum_candidates_per_source1": counts["maximum_candidates_per_source1"],
        "zero_candidate_source1_rows": counts["zero_candidate_source1_rows"],
        "zero_candidate_percentage": (
            100.0 * counts["zero_candidate_source1_rows"] / rows if rows else 0.0
        ),
        "predicted_matches": counts["predicted_matches"],
        "threshold": counts["threshold"],
        "model_strategy": counts["model_strategy"],
    }


def write_v4_test_predictions(
    project_root,
    output_dir=None,
    max_candidates_per_entity=100,
    batch_pairs=25000,
):
    """Generate V4 final candidates and matches from the saved V4 models.

    Candidate IDs written to candidate_pairs.tsv are exactly the IDs whose feature
    vectors are scored. For a country with no trained model, candidate scores are
    the arithmetic mean of the trained V4 country-model probabilities and the
    threshold is the arithmetic mean of their validation-selected thresholds.
    """
    config = V4Config(
        Path(project_root), max_candidates_per_entity=max_candidates_per_entity
    )
    root = Path(project_root)
    data_dir = config.data_dir / "test"
    source1_path = data_dir / "test_source1.tsv"
    source2_path = data_dir / "test_source2.tsv"
    source3_path = data_dir / "test_source3.tsv"
    for path in (source1_path, source2_path, source3_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing test source file: {path}")
    if batch_pairs < 1:
        raise ValueError("batch_pairs must be positive")

    artifact_dir = config.artifacts_dir
    models, thresholds, fallback_threshold, np, validation_report = _load_models(artifact_dir)
    countries = defaultdict(int)
    for record in iter_source(source1_path, "S1"):
        countries[normalize_country(record.country)] += 1
    country_list = sorted(countries)
    total_s1_rows = sum(countries.values())
    print(
        f"V4 inference: {total_s1_rows:,} S1 rows across {len(country_list)} countries; "
        f"cap={max_candidates_per_entity}; model countries={sorted(models)}; "
        f"unseen-country threshold={fallback_threshold:.4f}",
        flush=True,
    )

    if output_dir is None:
        output_dir = root / "final_submission" / "output"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_tmp = output_dir / "matching_results.tsv.tmp"
    candidate_tmp = output_dir / "candidate_pairs.tsv.tmp"

    overall_counts = {
        "source1_rows": 0,
        "candidate_pairs": 0,
        "maximum_candidates_per_source1": 0,
        "zero_candidate_source1_rows": 0,
        "predicted_matches": 0,
        "candidate_histogram": [0] * (max_candidates_per_entity + 1),
    }
    country_reports = {}
    inference_started = perf_counter()

    with matching_tmp.open("w", encoding="utf-8", newline="") as match_handle, candidate_tmp.open(
        "w", encoding="utf-8", newline=""
    ) as candidate_handle:
        match_writer = csv.writer(match_handle, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_handle, delimiter="\t", lineterminator="\n")
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for country in country_list:
            print(
                f"Indexing test S2/S3 records for country={country!r} "
                f"({countries[country]:,} S1 rows)...",
                flush=True,
            )
            index = V4TargetIndex(
                country,
                max_token_posting=config.max_token_posting,
                max_candidates=max_candidates_per_entity,
                v3_candidate_cap=50,
            ).extend_from_files(source2_path, source3_path)
            known_country = country in models
            threshold = thresholds[country] if known_country else fallback_threshold
            selected_models = [models[country]] if known_country else list(models.values())
            model_strategy = (
                "country_model" if known_country else "mean_probability_of_trained_country_models"
            )
            print(
                f"Indexed {len(index):,} target records for {country!r}; "
                f"threshold={threshold:.4f}; scoring={model_strategy}",
                flush=True,
            )

            counts = {
                "source1_rows": 0,
                "target_records_indexed": len(index),
                "candidate_pairs": 0,
                "maximum_candidates_per_source1": 0,
                "zero_candidate_source1_rows": 0,
                "predicted_matches": 0,
                "candidate_histogram": [0] * (max_candidates_per_entity + 1),
                "threshold": threshold,
                "model_strategy": model_strategy,
            }
            batch_rows = []
            flat_features = array.array("f")
            batch_candidate_count = 0

            def flush_batch():
                nonlocal flat_features, batch_candidate_count
                if not batch_rows:
                    return
                if batch_candidate_count:
                    matrix = np.frombuffer(flat_features, dtype=np.float32).reshape(
                        batch_candidate_count, len(V2_FEATURE_NAMES)
                    )
                    probabilities_by_model = [
                        model.predict_proba(matrix)[:, 1]
                        for model in selected_models
                    ]
                    if len(probabilities_by_model) == 1:
                        probabilities = probabilities_by_model[0]
                    else:
                        probabilities = np.mean(
                            np.vstack(probabilities_by_model), axis=0
                        )
                else:
                    probabilities = ()

                offset = 0
                for row in batch_rows:
                    size = len(row["candidate_ids"])
                    row_scores = probabilities[offset:offset + size]
                    offset += size
                    matches = [
                        entity_id for entity_id, score in zip(row["candidate_ids"], row_scores)
                        if float(score) >= threshold
                    ]
                    match_writer.writerow([row["s1_id"], ",".join(matches)])
                    candidate_writer.writerow([row["s1_id"], ",".join(row["candidate_ids"])])
                    candidate_count = size
                    counts["source1_rows"] += 1
                    counts["candidate_pairs"] += candidate_count
                    counts["candidate_histogram"][candidate_count] += 1
                    counts["maximum_candidates_per_source1"] = max(
                        counts["maximum_candidates_per_source1"], candidate_count
                    )
                    if not candidate_count:
                        counts["zero_candidate_source1_rows"] += 1
                    counts["predicted_matches"] += len(matches)

                if offset != batch_candidate_count:
                    raise AssertionError("Candidate feature scores lost row alignment")
                batch_rows.clear()
                flat_features = array.array("f")
                batch_candidate_count = 0

            for source1 in iter_source(source1_path, "S1", country):
                generated = index.candidates_with_sources(source1)
                candidate_ids = generated["candidate_ids"]
                candidate_indices = generated["candidate_indices"]
                if len(candidate_ids) != len(candidate_indices):
                    raise AssertionError("V4 candidate IDs and index positions are misaligned")
                if len(candidate_ids) > max_candidates_per_entity:
                    raise AssertionError("V4 candidate cap was exceeded")
                if len(candidate_ids) != len(set(candidate_ids)):
                    raise AssertionError("V4 emitted duplicate target entity IDs")

                prepared = prepare_source1_v2(source1)
                for idx in candidate_indices:
                    features = pair_features_v2(prepared, index.feature_record(idx))
                    if len(features) != len(V2_FEATURE_NAMES) or not all(
                        math.isfinite(float(value)) for value in features
                    ):
                        raise AssertionError("V4 inference feature count differs from the trained 29")
                    flat_features.extend(float(value) for value in features)
                batch_rows.append({"s1_id": source1.entity_id, "candidate_ids": candidate_ids})
                batch_candidate_count += len(candidate_ids)

                if (
                    len(batch_rows) >= 500
                    or batch_candidate_count >= batch_pairs
                ):
                    flush_batch()
                if (counts["source1_rows"] + len(batch_rows)) % 10000 == 0:
                    print(
                        f"  Prepared {counts['source1_rows'] + len(batch_rows):,}/"
                        f"{countries[country]:,} {country!r} S1 rows",
                        flush=True,
                    )
            flush_batch()
            if counts["source1_rows"] != countries[country]:
                raise AssertionError(
                    f"Processed {counts['source1_rows']} of {countries[country]} "
                    f"S1 rows for {country!r}"
                )

            summary = _country_summary(counts)
            country_reports[country] = summary
            print(
                f"Completed {country!r}: {summary['source1_rows']:,} S1, "
                f"{summary['candidate_pairs']:,} candidates, "
                f"{summary['predicted_matches']:,} predicted matches",
                flush=True,
            )
            for key in (
                "source1_rows", "candidate_pairs", "zero_candidate_source1_rows", "predicted_matches"
            ):
                overall_counts[key] += counts[key]
            overall_counts["maximum_candidates_per_source1"] = max(
                overall_counts["maximum_candidates_per_source1"],
                counts["maximum_candidates_per_source1"],
            )
            for size, frequency in enumerate(counts["candidate_histogram"]):
                overall_counts["candidate_histogram"][size] += frequency
            # Each country index contains millions of target-side postings; release
            # it before constructing the next country's index.
            del index
            gc.collect()

    if overall_counts["source1_rows"] != total_s1_rows:
        raise AssertionError(
            f"Generated {overall_counts['source1_rows']} S1 output rows; "
            f"test data has {total_s1_rows}"
        )

    os.replace(matching_tmp, matching_path)
    os.replace(candidate_tmp, candidate_path)
    overall_summary = _country_summary({
        **overall_counts,
        "target_records_indexed": 0,
        "threshold": None,
        "model_strategy": "country model or deterministic unseen-country fallback",
    })
    result = {
        "status": "PASS",
        "version": "V4",
        "candidate_cap": max_candidates_per_entity,
        "source1_rows": total_s1_rows,
        "matching_output_rows": overall_counts["source1_rows"],
        "candidate_output_rows": overall_counts["source1_rows"],
        "candidate_statistics": overall_summary,
        "predicted_matches": overall_counts["predicted_matches"],
        "predicted_matches_by_country": {
            country: details["predicted_matches"]
            for country, details in country_reports.items()
        },
        "countries": country_reports,
        "country_models": sorted(models),
        "thresholds_by_country": thresholds,
        "unseen_country_threshold": fallback_threshold,
        "unseen_country_scoring": "arithmetic mean of saved trained V4 model probabilities",
        "outputs": {
            "matching_results": os.path.relpath(matching_path.resolve(), root.resolve()),
            "candidate_pairs": os.path.relpath(candidate_path.resolve(), root.resolve()),
        },
        "validation_metrics_reference": validation_report.get("v4", {}),
        "runtime_seconds": round(perf_counter() - inference_started, 2),
    }
    report_path = artifact_dir / "v4_test_inference_report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result
