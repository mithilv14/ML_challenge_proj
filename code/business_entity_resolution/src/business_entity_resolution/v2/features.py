from functools import lru_cache

from ..features import FEATURE_NAMES as BASE_FEATURE_NAMES
from ..features import _char_grams, _jaccard, _token_set, pair_features
from ..normalize import normalize_address, normalize_name, tokens
from .normalization import (
    extract_identifiers_v2,
    normalize_address_v2,
    normalize_name_v2,
    strip_legal_suffix_v2,
)


V2_EXTRA_FEATURE_NAMES = (
    "name_core_exact", "name_token_prefix3_overlap", "name_char_4gram_jaccard",
    "address_char_4gram_jaccard", "email_exact", "email_conflict", "phone_exact",
    "phone_conflict", "postal_exact", "postal_conflict",
)
V2_FEATURE_NAMES = BASE_FEATURE_NAMES + V2_EXTRA_FEATURE_NAMES


@lru_cache(maxsize=200000)
def _prefix_tokens(value):
    return frozenset(token[:3] for token in tokens(value) if len(token) >= 5)


def prepare_source1_v2(record):
    base = (
        normalize_name(record.business_name), record.entity_id,
        normalize_address(record.business_address), record.source,
    )
    return {
        "base": base,
        "name": normalize_name_v2(record.business_name),
        "address": normalize_address_v2(record.business_address),
        "identifiers": extract_identifiers_v2(
            record.business_name, record.business_address, record.country
        ),
    }


def _agreement(left, right):
    return bool(set(left) & set(right))


def _conflict(left, right):
    return bool(left and right and not (set(left) & set(right)))


def pair_features_v2(source1, target):
    """Baseline similarities plus field-specific, conservative V2 signals."""
    base, identifiers, name2, address2 = target
    source_base = source1["base"]
    features = pair_features(source_base, base, True)

    name1, name2 = source1["name"], name2
    address1, address2 = source1["address"], address2
    name_prefixes_1, name_prefixes_2 = _prefix_tokens(name1), _prefix_tokens(name2)
    if name_prefixes_1 and name_prefixes_2:
        prefix_overlap = len(name_prefixes_1 & name_prefixes_2) / min(
            len(name_prefixes_1), len(name_prefixes_2)
        )
    else:
        prefix_overlap = 0.0

    source_ids = source1["identifiers"]
    target_emails = identifiers["emails"]
    target_phones = identifiers["phones"]
    target_postals = identifiers["postals"]
    email_exact = _agreement(source_ids["emails"], target_emails)
    phone_exact = _agreement(source_ids["phones"], target_phones)
    postal_exact = _agreement(source_ids["postals"], target_postals)
    features.extend([
        float(bool(strip_legal_suffix_v2(name1)) and
              strip_legal_suffix_v2(name1) == strip_legal_suffix_v2(name2)),
        prefix_overlap,
        _jaccard(_char_grams(name1, 4), _char_grams(name2, 4)),
        _jaccard(_char_grams(address1, 4), _char_grams(address2, 4)),
        float(email_exact), float(_conflict(source_ids["emails"], target_emails)),
        float(phone_exact), float(_conflict(source_ids["phones"], target_phones)),
        float(postal_exact), float(_conflict(source_ids["postals"], target_postals)),
    ])
    return features
