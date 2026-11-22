#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.v3.pipeline import static_check, train_v3_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate isolated V3 XGBoost scorer on the unchanged V2 pipeline."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--max-train-per-country", type=int, default=10000)
    parser.add_argument("--max-validation-per-country", type=int, default=2000)
    parser.add_argument("--max-candidates-per-entity", type=int, default=50)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run static/shared-feature checks only; do not read training data, train, or evaluate validation.",
    )
    args = parser.parse_args()
    if args.check:
        print(json.dumps(static_check(args.project_root), ensure_ascii=False, indent=2))
        return

    config = PipelineConfig(
        project_root=args.project_root,
        max_train_per_country=args.max_train_per_country,
        max_validation_per_country=args.max_validation_per_country,
        max_candidates_per_entity=args.max_candidates_per_entity,
    )
    print(json.dumps(train_v3_pipeline(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
