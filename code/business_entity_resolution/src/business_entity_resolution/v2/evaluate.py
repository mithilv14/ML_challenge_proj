from ..evaluate import ValidationAccumulator


def evaluate_cached_rows(rows, thresholds, target_count):
    """Sweep all thresholds from cached candidate scores without reblocking or rescoring."""
    baseline = ValidationAccumulator(thresholds)
    v2 = ValidationAccumulator(thresholds)
    for row in rows:
        truth = row["truth_ids"]
        baseline.add(row["truth_ids"], row["baseline_candidate_ids"], row["baseline_scores"])
        v2.add(truth, row["v2_candidate_ids"], row["v2_scores"])
    return {
        "baseline": baseline.report(target_count=target_count),
        "v2": v2.report(target_count=target_count),
    }


def tuned_overall(country_reports):
    """Combine per-country best-threshold results on the entity-weighted macro scale."""
    total_entities = sum(report["entities"] for report in country_reports.values())
    if not total_entities:
        return {}
    total_tp = sum(report["true_positive"] for report in country_reports.values())
    total_predicted = sum(report["predicted_pairs"] for report in country_reports.values())
    total_true = sum(report["true_matches"] for report in country_reports.values())
    total_blocked = sum(report["blocked_true_matches"] for report in country_reports.values())
    return {
        "validation_entities": total_entities,
        "macro_f0_5_country_tuned": sum(
            report["macro_f0_5"] * report["entities"] for report in country_reports.values()
        ) / total_entities,
        "precision_micro_country_tuned": total_tp / total_predicted if total_predicted else 0.0,
        "recall_micro_country_tuned": total_tp / total_true if total_true else 0.0,
        "blocking_recall": total_blocked / total_true if total_true else 0.0,
        "candidate_pairs": sum(report["candidate_pairs"] for report in country_reports.values()),
        "average_candidates_per_entity": sum(
            report["candidate_pairs"] for report in country_reports.values()
        ) / total_entities,
        "maximum_candidates_per_entity": max(
            report["maximum_candidates_per_entity"] for report in country_reports.values()
        ),
        "predicted_pairs_country_tuned": total_predicted,
        "true_matches": total_true,
        "blocked_true_matches": total_blocked,
    }
