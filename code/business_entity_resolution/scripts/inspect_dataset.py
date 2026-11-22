#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path


def inspect(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        rows = 0
        missing = Counter()
        country_counts = Counter()
        match_counts = Counter()
        for row in reader:
            rows += 1
            for column in columns:
                if not (row.get(column) or "").strip():
                    missing[column] += 1
            if "country" in row:
                country_counts[row.get("country") or "<EMPTY>"] += 1
            if "matched_entity_ids" in row:
                match_counts[len([x for x in (row.get("matched_entity_ids") or "").split(",") if x])] += 1
    return rows, columns, missing, country_counts, match_counts


def main():
    parser = argparse.ArgumentParser(description="Stream a compact report over the challenge TSV files.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    paths = sorted((args.project_root / "dataset").rglob("*.tsv"))
    for path in paths:
        rows, columns, missing, countries, match_counts = inspect(path)
        missing_pct = {column: round(100 * missing[column] / rows, 3) for column in columns} if rows else {}
        print(f"{path.relative_to(args.project_root)}: rows={rows}; columns={columns}; types=string; blank_pct={missing_pct}")
        if countries:
            print(f"  countries={dict(countries)}")
        if match_counts:
            print(f"  matches_per_source1={dict(sorted(match_counts.items()))}")


if __name__ == "__main__":
    main()
