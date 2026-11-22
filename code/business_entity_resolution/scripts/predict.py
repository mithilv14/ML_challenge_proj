#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.pipeline import predict_pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate matching and candidate TSV outputs for test data.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-candidates-per-entity", type=int, default=50)
    parser.add_argument("--threshold", type=float, help="Override the validation-tuned threshold for all countries.")
    args = parser.parse_args()
    config = PipelineConfig(project_root=args.project_root, max_candidates_per_entity=args.max_candidates_per_entity)
    result = predict_pipeline(config, args.model, args.output_dir, args.threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
