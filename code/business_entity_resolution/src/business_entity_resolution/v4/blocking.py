from array import array

from ..blocking import _ADDRESS_STOP, tokens
from ..data import iter_source
from ..normalize import normalize_name
from ..v2.blocking import V2TargetIndex
from ..v2.normalization import (
    _LEGAL_SUFFIXES,
    _SUFFIX_COLLAPSES,
    normalize_address_v2,
    normalize_name_v2,
)


_SUFFIX_FORMS = {}
for _form, _canonical in _SUFFIX_COLLAPSES:
    if len(_canonical) == 1:
        _SUFFIX_FORMS.setdefault(_canonical[0], []).append(_form)


class V4TargetIndex:
    """V3/V2 blocker plus bounded complementary name/address prefix proposals."""

    def __init__(
        self,
        country,
        max_token_posting=5000,
        max_candidates=100,
        v3_candidate_cap=50,
        address_prefix_posting_cap=25,
        exact_name_posting_cap=200,
        core_name_posting_cap=100,
    ):
        if max_candidates not in (75, 100):
            raise ValueError("V4 supports candidate caps of 75 or 100")
        self.country = country
        self.max_candidates = max_candidates
        self.v3_candidate_cap = v3_candidate_cap
        self.address_prefix_posting_cap = address_prefix_posting_cap
        self.exact_name_posting_cap = exact_name_posting_cap
        self.core_name_posting_cap = core_name_posting_cap
        self.v3_index = V2TargetIndex(
            country,
            max_token_posting=max_token_posting,
            max_candidates=max_candidates,
        )
        self.address_prefixes = {}
        self.address_prefix_overflow = set()

    def __len__(self):
        return len(self.v3_index)

    def add(self, record):
        idx = len(self.v3_index)
        self.v3_index.add(record)
        address = normalize_address_v2(record.business_address)
        for token in set(tokens(address)):
            if len(token) < 6 or token in _ADDRESS_STOP:
                continue
            self._bounded_add(
                self.address_prefixes,
                self.address_prefix_overflow,
                token[:4],
                idx,
                self.address_prefix_posting_cap,
            )

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

    def extend_from_files(self, source2, source3):
        for path, source in ((source2, "S2"), (source3, "S3")):
            for record in iter_source(path, source, self.country):
                self.add(record)
        return self

    def __getattr__(self, name):
        # Delegate V2 feature records and identifiers to the unchanged V3 index.
        return getattr(self.v3_index, name)

    def _register(self, proposals, strategy_ids, idx, strategy, score):
        entity_id = self.v3_index.base.ids[idx]
        strategy_ids[strategy].add(entity_id)
        entry = proposals.get(entity_id)
        if entry is None:
            entry = {"idx": idx, "strategy_scores": {}}
            proposals[entity_id] = entry
        entry["strategy_scores"][strategy] = max(
            score, entry["strategy_scores"].get(strategy, 0)
        )

    def _suffix_name_proposals(self, source1, proposals, strategy_ids):
        parts = normalize_name_v2(source1.business_name).split()
        if len(parts) < 2 or parts[-1] not in _LEGAL_SUFFIXES:
            return
        core = " ".join(parts[:-1])
        if len(core) < 3:
            return

        # The bare-core lookup recovers pairs where one source omitted a legal form.
        core_posting = self.v3_index.base.name_exact.get(core)
        if core_posting is not None and len(core_posting) <= self.core_name_posting_cap:
            for idx in core_posting:
                self._register(proposals, strategy_ids, idx, "name_core_exact", 4000)

        # Query only legal-form spellings that V2 itself collapses to this same suffix.
        forms = _SUFFIX_FORMS.get(parts[-1], ())
        for form in forms:
            key = normalize_name(f"{core} {' '.join(form)}")
            posting = self.v3_index.base.name_exact.get(key)
            if posting is None or len(posting) > self.exact_name_posting_cap:
                continue
            for idx in posting:
                self._register(proposals, strategy_ids, idx, "name_suffix_equivalent", 5000)

    def candidates_with_sources(self, source1):
        """Return unique entity IDs, aligned target indices, and pre-cap diagnostics."""
        v3_ranked, _, _ = self.v3_index.candidates_with_comparison(source1)
        v3_raw = v3_ranked[:self.v3_candidate_cap]
        v3_seed_ids = []
        v3_seed_indices = []
        seed_seen = set()
        for idx in v3_raw:
            entity_id = self.v3_index.base.ids[idx]
            if entity_id in seed_seen:
                continue
            seed_seen.add(entity_id)
            v3_seed_ids.append(entity_id)
            v3_seed_indices.append(idx)

        proposals = {}
        strategy_ids = {
            "v3_cap_expansion": set(),
            "name_suffix_equivalent": set(),
            "name_core_exact": set(),
            "single_name_prefix": set(),
            "address_prefix": set(),
        }

        # Preserve V3-ranked candidates 51..cap as an expansion source.
        for rank, idx in enumerate(v3_ranked[self.v3_candidate_cap:], start=1):
            self._register(
                proposals,
                strategy_ids,
                idx,
                "v3_cap_expansion",
                max(1, 1000 - rank),
            )

        self._suffix_name_proposals(source1, proposals, strategy_ids)

        source_name_tokens = {
            token for token in tokens(normalize_name_v2(source1.business_name))
            if len(token) >= 5
        }
        name_prefix_evidence = {}
        for token in source_name_tokens:
            posting = self.v3_index.name_prefixes.get(token[:3])
            if not posting or len(posting) > 50:
                continue
            for idx in posting:
                entity_id = self.v3_index.base.ids[idx]
                evidence = name_prefix_evidence.setdefault(entity_id, [idx, set(), 51])
                evidence[1].add(token[:3])
                evidence[2] = min(evidence[2], len(posting))
        for entity_id, (idx, prefix_keys, posting_size) in name_prefix_evidence.items():
            # V3 admitted a lone 3-char prefix only for postings of at most five.
            # V4 adds bounded one-prefix candidates for postings up to fifty.
            if entity_id not in seed_seen:
                score = 1200 + min(5, len(prefix_keys)) * 100 + (50 - posting_size)
                self._register(proposals, strategy_ids, idx, "single_name_prefix", score)
            else:
                self._register(proposals, strategy_ids, idx, "single_name_prefix", 1200)

        address_tokens = {
            token for token in tokens(normalize_address_v2(source1.business_address))
            if len(token) >= 6 and token not in _ADDRESS_STOP
        }
        address_prefix_evidence = {}
        for token in address_tokens:
            posting = self.address_prefixes.get(token[:4])
            if not posting or len(posting) > self.address_prefix_posting_cap:
                continue
            for idx in posting:
                entity_id = self.v3_index.base.ids[idx]
                evidence = address_prefix_evidence.setdefault(entity_id, [idx, set(), 26])
                evidence[1].add(token[:4])
                evidence[2] = min(evidence[2], len(posting))
        for entity_id, (idx, prefix_keys, posting_size) in address_prefix_evidence.items():
            score = 700 + min(5, len(prefix_keys)) * 100 + (25 - posting_size)
            if entity_id in seed_seen:
                score = 700
            self._register(proposals, strategy_ids, idx, "address_prefix", score)

        all_ids = seed_seen | set(proposals)
        ranked_addons = sorted(
            (entity_id for entity_id in proposals if entity_id not in seed_seen),
            key=lambda entity_id: (
                -sum(proposals[entity_id]["strategy_scores"].values()),
                entity_id,
            ),
        )
        selected_addons = ranked_addons[:max(0, self.max_candidates - len(v3_seed_ids))]
        selected_ids = v3_seed_ids + selected_addons
        index_by_id = {entity_id: idx for entity_id, idx in zip(v3_seed_ids, v3_seed_indices)}
        index_by_id.update({entity_id: proposals[entity_id]["idx"] for entity_id in selected_addons})
        raw_proposals = len(v3_raw) + len(v3_ranked[self.v3_candidate_cap:]) + sum(
            len(values) for values in strategy_ids.values()
        )
        stats = {
            "v3_candidates_at_50": len(v3_seed_ids),
            "v3_candidates_available_to_cap": len({self.v3_index.base.ids[idx] for idx in v3_ranked}),
            "v4_unique_candidates_before_cap": len(all_ids),
            "v4_candidates_after_cap": len(selected_ids),
            "v4_candidates_added_beyond_v3_50": len(set(selected_ids) - seed_seen),
            "candidate_proposals_deduplicated": max(0, raw_proposals - len(all_ids)),
            "candidate_cap_applied": len(all_ids) > self.max_candidates,
            "candidate_cap": self.max_candidates,
            "strategy_unique_candidate_counts": {
                name: len(ids) for name, ids in strategy_ids.items()
            },
        }
        return {
            "v3_candidate_ids": v3_seed_ids,
            "v3_candidate_indices": v3_seed_indices,
            "candidate_ids": selected_ids,
            "candidate_indices": [index_by_id[entity_id] for entity_id in selected_ids],
            "stats": stats,
        }
