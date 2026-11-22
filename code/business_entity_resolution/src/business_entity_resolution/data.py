import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence

from .normalize import normalize_country


SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


class Record(NamedTuple):
    entity_id: str
    business_name: str
    business_address: str
    country: str
    source: str


def _validate_header(path: Path, expected: Sequence[str], reader: csv.DictReader) -> None:
    if tuple(reader.fieldnames or ()) != tuple(expected):
        raise ValueError(
            f"Unexpected columns in {path}: {reader.fieldnames!r}; expected {list(expected)!r}"
        )


def iter_source(path: Path, source: str, country: Optional[str] = None) -> Iterator[Record]:
    """Stream a source TSV. All TSV reads explicitly use a tab delimiter."""
    wanted_country = normalize_country(country) if country is not None else None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _validate_header(path, SOURCE_COLUMNS, reader)
        for row in reader:
            row_country = row.get("country") or ""
            if wanted_country is not None and normalize_country(row_country) != wanted_country:
                continue
            yield Record(
                entity_id=row.get("entity_id") or "",
                business_name=row.get("business_name") or "",
                business_address=row.get("business_address") or "",
                country=row_country,
                source=source,
            )


def iter_ground_truth(path: Path) -> Iterator[tuple]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _validate_header(path, GROUND_TRUTH_COLUMNS, reader)
        for row in reader:
            raw = (row.get("matched_entity_ids") or "").strip()
            ids = tuple(value for value in raw.split(",") if value)
            yield row.get("source1_entity_id") or "", ids


def load_ground_truth(path: Path, wanted_ids: Optional[set] = None) -> Dict[str, set]:
    """Load parsed match lists, optionally retaining only selected S1 IDs."""
    labels: Dict[str, set] = {}
    for entity_id, matches in iter_ground_truth(path):
        if wanted_ids is None or entity_id in wanted_ids:
            labels[entity_id] = set(matches)
    return labels


def list_countries(path: Path, source: str) -> List[str]:
    countries = set()
    for record in iter_source(path, source):
        if normalize_country(record.country):
            countries.add(normalize_country(record.country))
    return sorted(countries)

