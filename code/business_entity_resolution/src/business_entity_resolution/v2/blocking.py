from array import array
from pathlib import Path

from ..blocking import TargetIndex
from ..data import iter_source
from ..normalize import normalize_address, normalize_name, tokens
from .normalization import extract_identifiers_v2, normalize_address_v2, normalize_name_v2


class V2TargetIndex:
    """Baseline token index plus bounded prefix and exact-identifier postings."""

    def __init__(self, country, max_token_posting=5000, max_candidates=50):
        self.country = country
        self.max_candidates = max_candidates
        self.base = TargetIndex(
            country,
            max_token_posting=max_token_posting,
            max_candidates=max_candidates,
        )
        self.v2_names = []
        self.v2_addresses = []
        self.identifiers = []
        self.name_prefixes = {}
        self.name_prefix_overflow = set()
        self.identifier_indices = {field: {} for field in ("emails", "phones", "postals")}
        self.identifier_overflow = {field: set() for field in ("emails", "phones", "postals")}
        self._posting_limits = {"emails": 500, "phones": 500, "postals": 250}

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _bounded_add(index, overflow, key, idx, limit):
        if key in overflow:
            return
        posting = index.get(key)
        if posting is None:
            posting = array("I")
            index[key] = posting
        if len(posting) >= limit:
            del index[key]
            overflow.add(key)
            return
        posting.append(idx)

    def add(self, record):
        idx = len(self.base)
        self.base.add(record)
        name = normalize_name_v2(record.business_name)
        address = normalize_address_v2(record.business_address)
        identifiers = extract_identifiers_v2(
            record.business_name, record.business_address, record.country
        )
        self.v2_names.append(name)
        self.v2_addresses.append(address)
        self.identifiers.append(identifiers)

        for token in set(tokens(name)):
            if len(token) >= 5:
                key = token[:3]
                self._bounded_add(self.name_prefixes, self.name_prefix_overflow, key, idx, 50)
        for field, values in identifiers.items():
            index = self.identifier_indices[field]
            overflow = self.identifier_overflow[field]
            limit = self._posting_limits[field]
            for value in values:
                self._bounded_add(index, overflow, value, idx, limit)

    def extend_from_files(self, source2: Path, source3: Path):
        for path, source in ((source2, "S2"), (source3, "S3")):
            for record in iter_source(path, source, self.country):
                self.add(record)
        return self

    def record(self, idx):
        return self.base.record(idx)

    def feature_record(self, idx):
        return (
            self.base.record(idx),
            self.identifiers[idx],
            self.v2_names[idx],
            self.v2_addresses[idx],
        )

    def candidates_with_comparison(self, source1):
        """Return capped V2 and baseline candidates from one pass over shared postings."""
        name = normalize_name(source1.business_name)
        address = normalize_address(source1.business_address)
        name_tokens = set(tokens(name))
        address_tokens = set(tokens(address))
        v2_name_tokens = {token for token in tokens(normalize_name_v2(source1.business_name))
                          if len(token) >= 5}
        support = {}
        exact_name = set()

        # Per-candidate evidence: name hits, address hits, rare name/address,
        # fuzzy prefix hits, rare prefix, email, phone, postal and postal rarity.
        def counts_for(idx):
            result = support.get(idx)
            if result is None:
                result = [0, 0, False, False, 0, False, False, False, False, 0]
                support[idx] = result
            return result

        exact_posting = self.base.name_exact.get(name) if len(name) >= 3 else None
        if exact_posting is not None and len(exact_posting) <= 200:
            exact_name.update(exact_posting)

        def add_token_support(posting, field):
            if posting is None or len(posting) > self.base.max_token_posting:
                return
            rare = len(posting) <= 100
            for idx in posting:
                counts = counts_for(idx)
                counts[field] += 1
                if rare:
                    counts[2 + field] = True

        for token in name_tokens:
            if len(token) >= 2:
                add_token_support(self.base.name_tokens.get(token), 0)
        for token in address_tokens:
            if len(token) >= 2:
                add_token_support(self.base.address_tokens.get(token), 1)

        for token in v2_name_tokens:
            key = token[:3]
            posting = self.name_prefixes.get(key)
            if posting is None or len(posting) > 50:
                continue
            rare = len(posting) <= 5
            for idx in posting:
                counts = counts_for(idx)
                counts[4] += 1
                counts[5] = counts[5] or rare

        source_ids = extract_identifiers_v2(
            source1.business_name, source1.business_address, source1.country
        )
        for field, slot in (("emails", 6), ("phones", 7), ("postals", 8)):
            index = self.identifier_indices[field]
            overflow = self.identifier_overflow[field]
            for value in source_ids[field]:
                if value in overflow:
                    continue
                posting = index.get(value)
                if posting is None:
                    continue
                for idx in posting:
                    counts = counts_for(idx)
                    counts[slot] = True
                    if field == "postals":
                        counts[9] = max(counts[9], len(posting))

        baseline_found = set(exact_name)
        v2_found = set(exact_name)
        for idx, evidence in support.items():
            name_hits, address_hits, rare_name, rare_address = evidence[:4]
            prefix_hits, rare_prefix, email_hit, phone_hit, postal_hit, postal_size = evidence[4:]
            baseline_eligible = (
                name_hits >= 2 or address_hits >= 2 or (name_hits and address_hits)
                or rare_name or rare_address
            )
            if baseline_eligible or idx in exact_name:
                baseline_found.add(idx)
                v2_found.add(idx)

            prefix_eligible = (
                prefix_hits >= 2 or (prefix_hits and (name_hits or address_hits))
                or (prefix_hits and rare_prefix)
            )
            postal_eligible = postal_hit and (
                postal_size <= 20 or name_hits or address_hits or prefix_hits
            )
            if email_hit or phone_hit or prefix_eligible or postal_eligible:
                v2_found.add(idx)

        def baseline_rank(idx):
            name_hits, address_hits, rare_name, rare_address = support.get(
                idx, (0, 0, False, False, 0, False, False, False, False, 0)
            )[:4]
            return (
                10 * (idx in exact_name) + 2 * name_hits + 2 * address_hits
                + 4 * rare_name + 3 * rare_address + 3 * bool(name_hits and address_hits)
            )

        def v2_rank(idx):
            evidence = support.get(idx, (0, 0, False, False, 0, False, False, False, False, 0))
            name_hits, address_hits, rare_name, rare_address = evidence[:4]
            prefix_hits, rare_prefix, email_hit, phone_hit, postal_hit, postal_size = evidence[4:]
            old_score = baseline_rank(idx)
            postal_boost = 8 if postal_hit and postal_size <= 3 else 4 if postal_hit and postal_size <= 20 else 2 if postal_hit else 0
            return (
                old_score + 60 * email_hit + 60 * phone_hit + postal_boost
                + 3 * prefix_hits + 4 * rare_prefix
            )

        baseline_ranked = sorted(baseline_found, key=lambda idx: (-baseline_rank(idx), self.base.ids[idx]))
        v2_ranked = sorted(v2_found, key=lambda idx: (-v2_rank(idx), self.base.ids[idx]))
        baseline_candidates = baseline_ranked[:self.max_candidates]
        v2_candidates = v2_ranked[:self.max_candidates]
        stats = {
            "baseline_pre_cap_candidates": len(baseline_found),
            "v2_pre_cap_candidates": len(v2_found),
            "additional_pre_cap_candidates": len(v2_found - baseline_found),
            "baseline_capped": len(baseline_found) > self.max_candidates,
            "v2_capped": len(v2_found) > self.max_candidates,
        }
        return v2_candidates, baseline_candidates, stats
