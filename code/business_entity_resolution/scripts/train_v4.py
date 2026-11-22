#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.v4.config import SUPPORTED_CANDIDATE_CAPS, V4Config
from business_entity_resolution.v4.pipeline import run_smoke_test, train_v4_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate isolated V4 blocking experiment with the unchanged V3 scorer."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--max-train-per-country", type=int, default=10000)
    parser.add_argument("--max-validation-per-country", type=int, default=2000)
    parser.add_argument(
        "--max-candidates-per-entity",
        type=int,
        choices=SUPPORTED_CANDIDATE_CAPS,
        default=100,
        help="V4 cap: choose 75 or 100 (default: 100).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run tiny synthetic blockers/features/schema/path checks only; never train or index full data.",
    )
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(run_smoke_test(args.project_root), ensure_ascii=False, indent=2))
        return

    config = V4Config(
        project_root=args.project_root,
        max_train_per_country=args.max_train_per_country,
        max_validation_per_country=args.max_validation_per_country,
        max_candidates_per_entity=args.max_candidates_per_entity,
    )
    print(json.dumps(train_v4_pipeline(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
