#!/usr/bin/env python3
"""Run V4 candidate generation and test inference using saved country models."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.v4.config import SUPPORTED_CANDIDATE_CAPS, V4Config
from business_entity_resolution.v4.inference import write_v4_test_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Generate final test candidates and matches using the trained V4 models."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Challenge root containing dataset/ (default: inferred from this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <project-root>/final_submission/output).",
    )
    parser.add_argument(
        "--max-candidates-per-entity",
        type=int,
        choices=SUPPORTED_CANDIDATE_CAPS,
        default=100,
    )
    parser.add_argument(
        "--batch-pairs",
        type=int,
        default=25000,
        help="Approximate maximum number of candidate feature rows per scoring batch.",
    )
    args = parser.parse_args()
    V4Config(args.project_root, max_candidates_per_entity=args.max_candidates_per_entity)
    result = write_v4_test_predictions(
        project_root=args.project_root,
        output_dir=args.output_dir,
        max_candidates_per_entity=args.max_candidates_per_entity,
        batch_pairs=args.batch_pairs,
    )
    print(json.dumps({
        "status": result["status"],
        "outputs": result["outputs"],
        "source1_rows": result["source1_rows"],
        "candidate_pairs": result["candidate_statistics"]["candidate_pairs"],
        "predicted_matches": result["predicted_matches"],
        "runtime_seconds": result["runtime_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
