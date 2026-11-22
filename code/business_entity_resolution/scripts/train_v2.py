#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.data import Record
from business_entity_resolution.model import LogisticRegressionSGD
from business_entity_resolution.v2.blocking import V2TargetIndex
from business_entity_resolution.v2.evaluate import evaluate_cached_rows
from business_entity_resolution.v2.features import (
    V2_FEATURE_NAMES,
    pair_features_v2,
    prepare_source1_v2,
)
from business_entity_resolution.v2.normalization import (
    extract_identifiers_v2,
    normalize_address_v2,
    normalize_name_v2,
)
from business_entity_resolution.v2.pipeline import train_v2_pipeline


def run_smoke_test():
    if normalize_name_v2("Northstar Incorporated") != normalize_name_v2("NORTHSTAR INC."):
        raise AssertionError("Name case/legal suffix normalization failed")
    if normalize_address_v2("10 N. Main St.") != "10 north main street":
        raise AssertionError("Address direction/abbreviation normalization failed")

    identifiers = extract_identifiers_v2(
        "Different Grocery", "12 Oak Road; Tel +1 (212) 555-0100; support@demo.test", "US"
    )
    if "support@demo.test" not in identifiers["emails"] or "2125550100" not in identifiers["phones"]:
        raise AssertionError(f"Conservative contact extraction failed: {identifiers}")

    index = V2TargetIndex("us", max_candidates=50)
    target = Record(
        "S2-000000001", "Nortstar Advisory LLC", "901 Cedar Boulevard, Boston, MA", "US", "S2"
    )
    contact_target = Record(
        "S3-000000002", "Courier Overseas Ltd",
        "12 Oak Road; Tel 212-555-0100; support@demo.test", "US", "S3",
    )
    index.add(target)
    index.add(contact_target)

    fuzzy_source = Record(
        "S1-000000001", "Northstar Solutions Inc", "77 Pine Avenue, Denver, CO", "US", "S1"
    )
    v2_ids, baseline_ids, _ = index.candidates_with_comparison(fuzzy_source)
    fuzzy_idx = index.base.ids.index(target.entity_id)
    if fuzzy_idx not in v2_ids or fuzzy_idx in baseline_ids:
        raise AssertionError("Prefix blocking did not recover the isolated typo example")

    contact_source = Record(
        "S1-000000002", "Different Grocery", 
        "98 Birch Lane; Phone +1 212 555 0100; SUPPORT@DEMO.TEST", "US", "S1",
    )
    contact_v2_ids, contact_baseline_ids, _ = index.candidates_with_comparison(contact_source)
    contact_idx = index.base.ids.index(contact_target.entity_id)
    if contact_idx not in contact_v2_ids:
        raise AssertionError("Contact-bearing pair was not retained in the V2 candidate list")

    prepared = prepare_source1_v2(contact_source)
    pair = pair_features_v2(prepared, index.feature_record(contact_idx))
    if len(pair) != len(V2_FEATURE_NAMES):
        raise AssertionError(f"Feature count mismatch: {len(pair)} != {len(V2_FEATURE_NAMES)}")
    if pair[V2_FEATURE_NAMES.index("email_exact")] != 1.0 or pair[V2_FEATURE_NAMES.index("phone_exact")] != 1.0:
        raise AssertionError("Pair features did not preserve exact contact agreement")
    postal_identifiers = extract_identifiers_v2(
        "Different Grocery", "12 Oak Road, Boston MA 02108", "US"
    )
    if "02108" not in postal_identifiers["postals"]:
        raise AssertionError("Postal code extraction did not find the terminal US ZIP")

    model = LogisticRegressionSGD(len(V2_FEATURE_NAMES))
    model.update(pair, 1)
    score = model.predict_proba(pair)
    if not 0.0 <= score <= 1.0:
        raise AssertionError("V2 model produced an invalid probability")

    sample_cache = [{
        "truth_ids": [contact_target.entity_id],
        "baseline_candidate_ids": [],
        "baseline_scores": [],
        "v2_candidate_ids": [contact_target.entity_id],
        "v2_scores": [score],
    }]
    evaluated = evaluate_cached_rows(sample_cache, (0.5, 0.8), len(index))
    if evaluated["v2"]["candidate_pairs"] != 1:
        raise AssertionError("Cached threshold evaluation failed")
    print(json.dumps({
        "status": "PASS",
        "feature_count": len(V2_FEATURE_NAMES),
        "fuzzy_candidates_recovered": True,
        "exact_contact_candidates_recovered": True,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the isolated V2 entity-resolution experiment.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--max-train-per-country", type=int, default=10000)
    parser.add_argument("--max-validation-per-country", type=int, default=2000)
    parser.add_argument("--max-candidates-per-entity", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="Run tiny synthetic code-path checks; do not build full indexes or write artifacts.")
    args = parser.parse_args()
    if args.smoke:
        run_smoke_test()
        return

    config = PipelineConfig(
        project_root=args.project_root,
        max_train_per_country=args.max_train_per_country,
        max_validation_per_country=args.max_validation_per_country,
        max_candidates_per_entity=args.max_candidates_per_entity,
    )
    _, report = train_v2_pipeline(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
