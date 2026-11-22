#!/usr/bin/env python3
"""Validate and package final V4 test outputs with only required source/models."""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import zipfile
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.data import iter_source
from business_entity_resolution.normalize import normalize_country
from business_entity_resolution.v4.config import SUPPORTED_CANDIDATE_CAPS, V4Config


def _format(value, digits=6):
    return f"{float(value):.{digits}f}"


def _country_label(country):
    return {"us": "US", "usa": "USA", "france": "France", "india": "India"}.get(
        country, country.title() or "(blank)"
    )


def _validation_markdown(report):
    overall = report.get("v4", {})
    lines = [
        "| Scope | Blocking recall | Precision | Recall | Macro F0.5 | Threshold | Candidate pairs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| Overall, country-tuned | "
        + " | ".join([
            _format(overall.get("blocking_recall", 0)),
            _format(overall.get("precision_micro_country_tuned", overall.get("precision_micro", 0))),
            _format(overall.get("recall_micro_country_tuned", overall.get("recall_micro", 0))),
            _format(overall.get("macro_f0_5_country_tuned", overall.get("macro_f0_5", 0))),
            "per-country best",
            f"{int(overall.get('candidate_pairs', 0)):,}",
        ]) + " |",
    ]
    for country, details in sorted(report.get("country_reports", {}).items()):
        metrics = details.get("v4", {})
        lines.append(
            f"| {_country_label(country)} | {_format(metrics.get('blocking_recall', 0))} "
            f"| {_format(metrics.get('precision_micro', 0))} "
            f"| {_format(metrics.get('recall_micro', 0))} "
            f"| {_format(metrics.get('macro_f0_5', 0))} "
            f"| {_format(metrics.get('best_threshold', 0), 2)} "
            f"| {int(metrics.get('candidate_pairs', 0)):,} |"
        )
    lines.extend([
        "",
        f"Validation runtime: {float(report.get('runtime_minutes', 0)):.2f} minutes. "
        f"Candidate cap: {int(report.get('candidate_cap', 100))} per S1.",
    ])
    return "\n".join(lines)


def _test_outputs_markdown(inference):
    stats = inference["candidate_statistics"]
    lines = [
        f"Test S1/output rows: **{int(inference['source1_rows']):,}** in each file.",
        f"Total candidate pairs: **{int(stats['candidate_pairs']):,}**; "
        f"average **{float(stats['average_candidates_per_source1']):.3f}**, "
        f"median **{float(stats['median_candidates_per_source1']):g}**, "
        f"maximum **{int(stats['maximum_candidates_per_source1'])}**; "
        f"zero-candidate rows **{int(stats['zero_candidate_source1_rows']):,}** "
        f"({float(stats['zero_candidate_percentage']):.3f}%).",
        f"Predicted matches: **{int(inference['predicted_matches']):,}**.",
        "",
        "| Test country | S1 rows | Candidate pairs | Average candidates | Median | Maximum | Zero candidates | Predicted matches | Threshold | Scoring rule |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for country, details in sorted(inference["countries"].items()):
        lines.append(
            f"| {_country_label(country)} | {int(details['source1_rows']):,} "
            f"| {int(details['candidate_pairs']):,} "
            f"| {float(details['average_candidates_per_source1']):.3f} "
            f"| {float(details['median_candidates_per_source1']):g} "
            f"| {int(details['maximum_candidates_per_source1'])} "
            f"| {int(details['zero_candidate_source1_rows']):,} "
            f"| {int(details['predicted_matches']):,} "
            f"| {float(details['threshold']):.2f} "
            f"| {details['model_strategy']} |"
        )
    lines.extend([
        "",
        "These are inference volume counts, not test accuracy. Test labels were not used.",
    ])
    return "\n".join(lines)


def _render_documentation(template_path, inference, validation_report):
    text = template_path.read_text(encoding="utf-8")
    start_marker = "<!-- GENERATED_V4_VALIDATION_RESULTS_BEGIN -->"
    end_marker = "<!-- GENERATED_V4_VALIDATION_RESULTS_END -->"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise ValueError("Expected one bounded V4 validation-results section")
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker)
    if start > end:
        raise ValueError("V4 validation-results section markers are out of order")
    text = text[:start] + "\n\n" + _validation_markdown(validation_report) + "\n\n" + text[end:]
    marker = "<!-- GENERATED_FINAL_TEST_RESULTS -->"
    if text.count(marker) != 1:
        raise ValueError(f"Expected one documentation placeholder: {marker}")
    text = text.replace(marker, _test_outputs_markdown(inference))
    text = text.replace(start_marker, "").replace(end_marker, "")
    return text


def _median(histogram, count):
    if not count:
        return 0.0
    ranks = ((count - 1) // 2, count // 2)
    found, cumulative, position = [], 0, 0
    for value, frequency in enumerate(histogram):
        cumulative += frequency
        while position < len(ranks) and ranks[position] < cumulative:
            found.append(value)
            position += 1
    return sum(found) / len(found)


def _verify_output_files(matching_path, candidate_path, cap, inference):
    """Stream both result files and independently compare core counts/invariants."""
    candidate_pairs = match_count = rows = zero_candidates = maximum = 0
    histogram = [0] * (cap + 1)
    with matching_path.open("r", encoding="utf-8", newline="") as match_handle, candidate_path.open(
        "r", encoding="utf-8", newline=""
    ) as candidate_handle:
        match_reader = csv.reader(match_handle, delimiter="\t")
        candidate_reader = csv.reader(candidate_handle, delimiter="\t")
        if next(match_reader, None) != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("Unexpected matching_results.tsv header")
        if next(candidate_reader, None) != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError("Unexpected candidate_pairs.tsv header")
        sentinel = object()
        for match_row, candidate_row in zip_longest(match_reader, candidate_reader, fillvalue=sentinel):
            if match_row is sentinel or candidate_row is sentinel:
                raise ValueError("Matching and candidate files have different row counts")
            if len(match_row) != 2 or len(candidate_row) != 2:
                raise ValueError(f"Malformed output row near row {rows + 2}")
            if match_row[0] != candidate_row[0]:
                raise ValueError(f"Matching/candidate S1 row alignment differs at row {rows + 2}")
            candidate_ids = candidate_row[1].split(",") if candidate_row[1] else []
            matched_ids = match_row[1].split(",") if match_row[1] else []
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"Duplicate candidate IDs for {candidate_row[0]}")
            if len(matched_ids) != len(set(matched_ids)):
                raise ValueError(f"Duplicate matched IDs for {match_row[0]}")
            if not set(matched_ids).issubset(candidate_ids):
                raise ValueError(f"A prediction for {match_row[0]} is absent from its candidate list")
            if len(candidate_ids) > cap:
                raise ValueError(f"Candidate cap exceeded for {candidate_row[0]}")
            candidate_pairs += len(candidate_ids)
            match_count += len(matched_ids)
            histogram[len(candidate_ids)] += 1
            maximum = max(maximum, len(candidate_ids))
            zero_candidates += not candidate_ids
            rows += 1

    expected = inference["candidate_statistics"]
    comparisons = {
        "rows": (rows, int(inference["source1_rows"])),
        "candidate_pairs": (candidate_pairs, int(expected["candidate_pairs"])),
        "average_candidates": (
            candidate_pairs / rows if rows else 0.0,
            float(expected["average_candidates_per_source1"]),
        ),
        "predicted_matches": (match_count, int(inference["predicted_matches"])),
        "maximum_candidates": (maximum, int(expected["maximum_candidates_per_source1"])),
        "zero_candidate_rows": (zero_candidates, int(expected["zero_candidate_source1_rows"])),
        "median_candidates": (
            _median(histogram, rows), float(expected["median_candidates_per_source1"])
        ),
    }
    for name, (actual, recorded) in comparisons.items():
        if not math.isclose(float(actual), float(recorded), rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Output {name}={actual} differs from inference report={recorded}")
    return {name: actual for name, (actual, _) in comparisons.items()}


def _required_source_files(source_package):
    paths = [
        "__init__.py", "blocking.py", "data.py", "features.py", "normalize.py",
        "v2/__init__.py", "v2/blocking.py", "v2/features.py", "v2/normalization.py",
        "v4/__init__.py", "v4/blocking.py", "v4/config.py", "v4/inference.py",
    ]
    missing = [relative for relative in paths if not (source_package / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing V4 runtime source files: " + ", ".join(missing))
    return paths


def package_v4_submission(
    project_root,
    submission_dir=None,
    outputs_dir=None,
    zip_path=None,
    max_candidates_per_entity=100,
    check_ids=True,
):
    root = Path(project_root).resolve()
    config = V4Config(root, max_candidates_per_entity=max_candidates_per_entity)
    submission_dir = Path(submission_dir or root / "final_submission")
    if not submission_dir.is_absolute():
        submission_dir = root / submission_dir
    submission_dir = submission_dir.resolve()
    package_output_dir = submission_dir / "output"
    input_output_dir = Path(outputs_dir) if outputs_dir is not None else package_output_dir
    if not input_output_dir.is_absolute():
        input_output_dir = root / input_output_dir
    input_output_dir = input_output_dir.resolve()
    matching_path = input_output_dir / "matching_results.tsv"
    candidate_path = input_output_dir / "candidate_pairs.tsv"
    inference_path = config.artifacts_dir / "v4_test_inference_report.json"
    validation_path = config.artifacts_dir / "v4_validation_report.json"
    source_package = root / "code/business_entity_resolution/src/business_entity_resolution"
    predict_script = root / "code/business_entity_resolution/scripts/predict_v4.py"
    requirements_path = root / "code/business_entity_resolution/requirements.txt"
    readme_path = root / "code/business_entity_resolution/README.md"
    documentation_path = root / "Documentation_template.md"

    for path in (matching_path, candidate_path, inference_path, validation_path,
                 predict_script, requirements_path, readme_path, documentation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required final submission input is missing: {path}")
    submission_dir.mkdir(parents=True, exist_ok=True)
    input_is_package_output = input_output_dir == package_output_dir.resolve()
    allowed_initial_entries = {"output"} if input_is_package_output else set()
    unexpected_root_entries = {
        path.name for path in submission_dir.iterdir()
        if path.name not in allowed_initial_entries
    }
    if unexpected_root_entries:
        raise FileExistsError(
            "Submission directory must be clean before packaging; unexpected entries: "
            + ", ".join(sorted(unexpected_root_entries))
            + ". Choose a fresh --submission-dir."
        )
    expected_output_entries = {"matching_results.tsv", "candidate_pairs.tsv"}
    if input_is_package_output:
        output_entries = {path.name for path in input_output_dir.iterdir()}
        if output_entries != expected_output_entries:
            raise ValueError(
                f"Output directory must contain exactly {sorted(expected_output_entries)}; "
                f"found {sorted(output_entries)}"
            )
    elif package_output_dir.exists():
        raise FileExistsError(
            f"Package output directory already exists: {package_output_dir}. "
            "Choose a fresh --submission-dir."
        )

    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    validation_report = json.loads(validation_path.read_text(encoding="utf-8"))
    if inference.get("status") != "PASS":
        raise ValueError("V4 test inference report does not have status PASS")
    cap = int(inference.get("candidate_cap", -1))
    if cap != max_candidates_per_entity or cap not in SUPPORTED_CANDIDATE_CAPS:
        raise ValueError(f"Inference report cap {cap} does not match requested cap {max_candidates_per_entity}")
    if int(validation_report.get("candidate_cap", -1)) != cap:
        raise ValueError("V4 validation report candidate cap differs from final inference cap")

    test_s1 = root / "dataset/test/test_source1.tsv"
    expected_country_rows = {}
    for record in iter_source(test_s1, "S1"):
        country = normalize_country(record.country)
        expected_country_rows[country] = expected_country_rows.get(country, 0) + 1
    actual_country_rows = {
        country: int(details["source1_rows"])
        for country, details in inference.get("countries", {}).items()
    }
    if expected_country_rows != actual_country_rows:
        raise ValueError(
            "Inference country coverage differs from test S1: "
            f"expected={expected_country_rows}, got={actual_country_rows}"
        )
    stats = inference["candidate_statistics"]
    if int(inference["matching_output_rows"]) != int(inference["source1_rows"]):
        raise ValueError("Matching row count in inference report differs from test S1 count")
    if int(inference["candidate_output_rows"]) != int(inference["source1_rows"]):
        raise ValueError("Candidate row count in inference report differs from test S1 count")
    if int(stats["maximum_candidates_per_source1"]) > cap:
        raise ValueError("V4 candidate maximum exceeds its configured cap")
    verified_counts = _verify_output_files(matching_path, candidate_path, cap, inference)

    model_files = []
    for raw_country in validation_report.get("country_reports", {}):
        country = normalize_country(raw_country)
        model_path = config.artifacts_dir / f"v4_xgboost_{country}.json"
        if not model_path.is_file():
            raise FileNotFoundError(f"Required V4 model is missing: {model_path}")
        model_files.append(model_path)
    if not model_files:
        raise ValueError("No trained V4 country models are listed in the validation report")

    validator = root / "utils/validate_submission.py"
    if not validator.is_file():
        raise FileNotFoundError(f"Supplied official validator not found: {validator}")
    validator_command = [
        sys.executable, str(validator),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(root / "dataset/test"),
    ]
    if check_ids:
        validator_command.append("--check-ids")
    print("Running supplied validator:", " ".join(validator_command), flush=True)
    validation = subprocess.run(
        validator_command, cwd=root, text=True, capture_output=True, check=False
    )
    if validation.stdout:
        print(validation.stdout, end="", flush=True)
    if validation.stderr:
        print(validation.stderr, end="", file=sys.stderr, flush=True)
    if validation.returncode:
        raise RuntimeError(f"Official submission validator failed with exit code {validation.returncode}")
    if "have matched IDs not present in" in validation.stdout:
        raise RuntimeError("Official validator warned that predicted matches are absent from candidates")
    if check_ids and "WARNING:" in validation.stdout:
        raise RuntimeError("Official validator returned an unexpected warning during ID checks")

    if not input_is_package_output:
        package_output_dir.mkdir(parents=True)
        shutil.copy2(matching_path, package_output_dir / matching_path.name)
        shutil.copy2(candidate_path, package_output_dir / candidate_path.name)
        matching_path = package_output_dir / matching_path.name
        candidate_path = package_output_dir / candidate_path.name

    package_code = submission_dir / "code/business_entity_resolution"
    if package_code.exists():
        raise FileExistsError(
            f"Package code directory already exists: {package_code}. "
            "Choose a fresh --submission-dir to keep this packaging run clean."
        )
    package_code.mkdir(parents=True)
    package_src = package_code / "src/business_entity_resolution"
    source_paths = _required_source_files(source_package)
    for relative in source_paths:
        target = package_src / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_package / relative, target)

    shutil.copy2(readme_path, package_code / "README.md")
    shutil.copy2(requirements_path, package_code / "requirements.txt")
    package_scripts = package_code / "scripts"
    package_scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(predict_script, package_scripts / "predict_v4.py")
    rendered_docs = _render_documentation(documentation_path, inference, validation_report)
    (submission_dir / "Documentation_template.md").write_text(rendered_docs, encoding="utf-8")

    package_artifacts = package_code / "artifacts/v4" / f"cap_{cap}"
    package_artifacts.mkdir(parents=True, exist_ok=True)
    for path in model_files:
        shutil.copy2(path, package_artifacts / path.name)
    portable_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    for raw_country, details in portable_validation.get("country_reports", {}).items():
        country = normalize_country(raw_country)
        if "model_path" in details:
            details["model_path"] = f"artifacts/v4/cap_{cap}/v4_xgboost_{country}.json"
    (package_artifacts / "v4_validation_report.json").write_text(
        json.dumps(portable_validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    portable_inference = json.loads(inference_path.read_text(encoding="utf-8"))
    portable_inference["outputs"] = {
        "matching_results": "output/matching_results.tsv",
        "candidate_pairs": "output/candidate_pairs.tsv",
    }
    (package_artifacts / "v4_test_inference_report.json").write_text(
        json.dumps(portable_inference, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if any("cache" in path.name.lower() for path in package_artifacts.iterdir()):
        raise AssertionError("A cache artifact must not be included in the final package")

    zip_path = Path(zip_path or root / "student_resource_v4_submission.zip")
    if not zip_path.is_absolute():
        zip_path = root / zip_path
    zip_path = zip_path.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as archive:
        for path in sorted(submission_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(submission_dir).as_posix())
    with zipfile.ZipFile(temporary_zip, "r") as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"ZIP verification failed at {bad_file}")
    temporary_zip.replace(zip_path)

    tree = sorted(
        str(path.relative_to(submission_dir)) + ("/" if path.is_dir() else "")
        for path in submission_dir.rglob("*")
    )
    print("Final submission tree:")
    for item in tree:
        print("  " + item)
    print(f"ZIP size: {zip_path.stat().st_size:,} bytes ({zip_path.stat().st_size / (1024 ** 2):.2f} MiB)")
    print("ZIP entries:")
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            print("  " + name)
    print(json.dumps({
        "status": "PASS",
        "matching_results": str(matching_path),
        "candidate_pairs": str(candidate_path),
        "documentation": str(submission_dir / "Documentation_template.md"),
        "readme": str(package_code / "README.md"),
        "zip": str(zip_path),
        "matching_rows": inference["matching_output_rows"],
        "candidate_rows": inference["candidate_output_rows"],
        "candidate_pairs_total": stats["candidate_pairs"],
        "average_candidates": stats["average_candidates_per_source1"],
        "median_candidates": stats["median_candidates_per_source1"],
        "maximum_candidates": stats["maximum_candidates_per_source1"],
        "zero_candidate_rows": stats["zero_candidate_source1_rows"],
        "predicted_matches": inference["predicted_matches"],
        "predicted_matches_by_country": inference["predicted_matches_by_country"],
        "output_file_integrity": verified_counts,
        "validator": "PASS",
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Validate V4 test outputs and assemble the clean submission ZIP."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--submission-dir", type=Path, default=None)
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument(
        "--max-candidates-per-entity", type=int,
        choices=SUPPORTED_CANDIDATE_CAPS, default=100,
    )
    parser.add_argument(
        "--skip-id-check", action="store_true",
        help="Run the official format/subset validator without its memory-heavy target-ID existence pass.",
    )
    args = parser.parse_args()
    package_v4_submission(
        project_root=args.project_root,
        submission_dir=args.submission_dir,
        outputs_dir=args.outputs_dir,
        zip_path=args.zip_path,
        max_candidates_per_entity=args.max_candidates_per_entity,
        check_ids=not args.skip_id_check,
    )


if __name__ == "__main__":
    main()
