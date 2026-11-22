#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.pipeline import train_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train and validate the entity-resolution baseline.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--max-train-per-country", type=int, default=10000)
    parser.add_argument("--max-validation-per-country", type=int, default=2000)
    parser.add_argument("--max-candidates-per-entity", type=int, default=50)
    args = parser.parse_args()
    config = PipelineConfig(
        project_root=args.project_root,
        max_train_per_country=args.max_train_per_country,
        max_validation_per_country=args.max_validation_per_country,
        max_candidates_per_entity=args.max_candidates_per_entity,
    )
    _, report = train_pipeline(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
