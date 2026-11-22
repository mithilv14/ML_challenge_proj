import csv
from pathlib import Path

from .blocking import TargetIndex
from .data import iter_source
from .features import pair_features
from .model import LogisticRegressionSGD
from .normalize import normalize_country, normalize_name, normalize_address


def write_test_predictions(config, artifact, output_dir=None, threshold_override=None):
    root = config.project_root
    data = config.data_dir
    output_dir = Path(output_dir or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    test_s1 = data / "test" / "test_source1.tsv"
    test_s2 = data / "test" / "test_source2.tsv"
    test_s3 = data / "test" / "test_source3.tsv"
    countries = set()
    for record in iter_source(test_s1, "S1"):
        country = normalize_country(record.country)
        if country:
            countries.add(country)

    models = {country: LogisticRegressionSGD.from_dict(payload) for country, payload in artifact["models"].items()}
    fallback_model = LogisticRegressionSGD.from_dict(artifact["fallback_model"])
    thresholds = artifact["threshold_by_country"]
    fallback_threshold = float(artifact["fallback_threshold"])
    stats = {"entities": 0, "candidates": 0, "matches": 0, "countries": {}}

    with matching_path.open("w", encoding="utf-8", newline="") as match_handle, candidate_path.open(
        "w", encoding="utf-8", newline=""
    ) as candidate_handle:
        matching_writer = csv.writer(match_handle, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_handle, delimiter="\t", lineterminator="\n")
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for country in sorted(countries):
            print(f"Indexing test S2/S3 targets for country={country}", flush=True)
            index = TargetIndex(
                country,
                max_token_posting=config.max_token_posting,
                max_candidates=config.max_candidates_per_entity,
            ).extend_from_files(test_s2, test_s3)
            print(f"Indexed {len(index)} test targets for country={country}", flush=True)
            model = models.get(country, fallback_model)
            threshold = float(threshold_override if threshold_override is not None
                              else thresholds.get(country, fallback_threshold))
            country_counts = {"source1_entities": 0, "target_entities": len(index), "candidates": 0, "matches": 0}
            for source1 in iter_source(test_s1, "S1", country):
                candidate_indices = index.candidates(source1)
                candidate_ids = [index.ids[idx] for idx in candidate_indices]
                n1, a1 = normalize_name(source1.business_name), normalize_address(source1.business_address)
                row1 = (n1, source1.entity_id, a1, source1.source)
                scores = [
                    model.predict_proba(pair_features(row1, index.record(idx), True))
                    for idx in candidate_indices
                ]
                matches = [entity_id for entity_id, score in zip(candidate_ids, scores) if score >= threshold]
                matching_writer.writerow([source1.entity_id, ",".join(matches)])
                candidate_writer.writerow([source1.entity_id, ",".join(candidate_ids)])
                country_counts["source1_entities"] += 1
                country_counts["candidates"] += len(candidate_ids)
                country_counts["matches"] += len(matches)
                if country_counts["source1_entities"] % 10000 == 0:
                    print(f"  scored {country_counts['source1_entities']} S1 rows in {country}", flush=True)
            stats["entities"] += country_counts["source1_entities"]
            stats["candidates"] += country_counts["candidates"]
            stats["matches"] += country_counts["matches"]
            stats["countries"][country] = country_counts
            country_counts["average_candidates_per_entity"] = (
                country_counts["candidates"] / country_counts["source1_entities"]
                if country_counts["source1_entities"] else 0.0
            )

    stats["matching_path"] = str(matching_path)
    stats["candidate_path"] = str(candidate_path)
    return stats
