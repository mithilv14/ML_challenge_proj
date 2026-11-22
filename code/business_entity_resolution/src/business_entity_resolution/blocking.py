from array import array
from collections import defaultdict
from pathlib import Path

from .data import iter_source
from .normalize import normalize_address, normalize_country, normalize_name, tokens


_NAME_STOP = {
    "and", "the", "of", "inc", "corp", "ltd", "pvt", "llc", "llp", "co",
    "company", "limited", "private", "corporation", "incorporated", "group",
    "service", "services", "international", "global", "enterprises", "enterprise",
}
_ADDRESS_STOP = {
    "and", "the", "of", "road", "street", "avenue", "boulevard", "highway",
    "lane", "drive", "parkway", "suite", "apartment", "floor", "near", "building",
    "city", "state", "district", "county", "india", "usa", "us",
}


class TargetIndex:
    """Country-partitioned S2/S3 target index with compact integer postings."""

    def __init__(self, country: str, max_token_posting: int = 5000, max_candidates: int = 50):
        self.country = normalize_country(country)
        self.max_token_posting = max_token_posting
        self.max_candidates = max_candidates
        self.ids = []
        self.names = []
        self.addresses = []
        self.sources = []
        self.name_exact = defaultdict(lambda: array("I"))
        self.name_tokens = defaultdict(lambda: array("I"))
        self.address_tokens = defaultdict(lambda: array("I"))

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _posting_add(index, key, value):
        index[key].append(value)

    def add(self, record):
        idx = len(self.ids)
        name = normalize_name(record.business_name)
        address = normalize_address(record.business_address)
        self.ids.append(record.entity_id)
        self.names.append(name)
        self.addresses.append(address)
        self.sources.append(record.source)

        if len(name) >= 3:
            self._posting_add(self.name_exact, name, idx)
            for token in set(tokens(name)):
                if len(token) >= 2 and token not in _NAME_STOP:
                    self._posting_add(self.name_tokens, token, idx)
        for token in set(tokens(address)):
            if len(token) >= 2 and token not in _ADDRESS_STOP:
                self._posting_add(self.address_tokens, token, idx)

    def extend_from_files(self, source2: Path, source3: Path):
        for path, source in ((source2, "S2"), (source3, "S3")):
            for record in iter_source(path, source, self.country):
                self.add(record)
        return self

    def candidates(self, source1_record):
        name = normalize_name(source1_record.business_name)
        address = normalize_address(source1_record.business_address)
        support = {}
        exact_name = set()

        def add_support(posting, field, rare_limit=100):
            if posting is None or len(posting) > self.max_token_posting:
                return
            rare = len(posting) <= rare_limit
            for idx in posting:
                counts = support.get(idx)
                if counts is None:
                    counts = [0, 0, False, False]
                    support[idx] = counts
                counts[field] += 1
                if rare:
                    counts[2 + field] = True

        if len(name) >= 3:
            exact_posting = self.name_exact.get(name)
            if exact_posting is not None and len(exact_posting) <= 200:
                exact_name.update(exact_posting)
            for token in set(tokens(name)):
                if len(token) >= 2 and token not in _NAME_STOP:
                    add_support(self.name_tokens.get(token), 0)
        for token in set(tokens(address)):
            if len(token) >= 2 and token not in _ADDRESS_STOP:
                add_support(self.address_tokens.get(token), 1)

        found = exact_name
        for idx, (name_hits, address_hits, rare_name, rare_address) in support.items():
            if (name_hits >= 2 or address_hits >= 2 or (name_hits and address_hits)
                    or rare_name or rare_address):
                found.add(idx)
        def rank(idx):
            name_hits, address_hits, rare_name, rare_address = support.get(idx, (0, 0, False, False))
            return (
                10 * (idx in exact_name)
                + 2 * name_hits + 2 * address_hits
                + 4 * rare_name + 3 * rare_address
                + 3 * bool(name_hits and address_hits)
            )
        ranked = sorted(found, key=lambda idx: (-rank(idx), self.ids[idx]))
        return ranked[:self.max_candidates]

    def record(self, idx):
        return (self.ids[idx], self.names[idx], self.addresses[idx], self.sources[idx])
