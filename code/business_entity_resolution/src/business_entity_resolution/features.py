from functools import lru_cache

from .normalize import tokens


FEATURE_NAMES = (
    "name_exact", "name_char_similarity", "name_token_jaccard", "name_token_containment",
    "name_char_bigram_jaccard", "name_length_ratio", "name_prefix_ratio",
    "address_exact", "address_char_similarity", "address_token_jaccard",
    "address_token_containment", "address_length_ratio", "address_number_overlap",
    "country_equal", "source_is_s3", "name_missing", "address_missing",
    "name_address_similarity_mean", "name_address_similarity_product",
)


@lru_cache(maxsize=200000)
def _token_set(value):
    return frozenset(tokens(value))


@lru_cache(maxsize=300000)
def _char_grams(value, size):
    compact = value.replace(" ", "")
    if len(compact) < size:
        return frozenset((compact,)) if compact else frozenset()
    return frozenset(compact[i:i + size] for i in range(len(compact) - size + 1))


def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return _jaccard(_char_grams(a, 3), _char_grams(b, 3))


def _length_ratio(a: str, b: str) -> float:
    longest = max(len(a), len(b))
    return min(len(a), len(b)) / longest if longest else 1.0


def _token_scores(a: str, b: str):
    left, right = _token_set(a), _token_set(b)
    overlap = len(left & right)
    union = len(left | right)
    containment = overlap / min(len(left), len(right)) if left and right else 0.0
    return (overlap / union if union else 1.0), containment, left, right


def _bigram_jaccard(a: str, b: str) -> float:
    return _jaccard(_char_grams(a, 2), _char_grams(b, 2))


def _prefix_ratio(a: str, b: str) -> float:
    n = 0
    for left, right in zip(a, b):
        if left != right:
            break
        n += 1
    return n / min(len(a), len(b)) if a and b else 0.0


def _number_overlap(a_tokens: set, b_tokens: set) -> float:
    left = {token for token in a_tokens if any(ch.isdigit() for ch in token)}
    right = {token for token in b_tokens if any(ch.isdigit() for ch in token)}
    if not left and not right:
        return 0.0
    return len(left & right) / min(len(left), len(right)) if left and right else 0.0


def pair_features(source1, target, country_equal: bool) -> list:
    name1, name2 = source1[0], target[1]
    address1, address2 = source1[2], target[2]
    nt_jac, nt_contain, _, _ = _token_scores(name1, name2)
    at_jac, at_contain, at1, at2 = _token_scores(address1, address2)
    name_char = _ratio(name1, name2)
    address_char = _ratio(address1, address2)
    name_mean = (name_char + nt_jac) / 2.0
    address_mean = (address_char + at_jac) / 2.0
    return [
        float(bool(name1) and name1 == name2), name_char, nt_jac, nt_contain,
        _bigram_jaccard(name1, name2), _length_ratio(name1, name2), _prefix_ratio(name1, name2),
        float(bool(address1) and address1 == address2), address_char, at_jac,
        at_contain, _length_ratio(address1, address2), _number_overlap(at1, at2),
        float(country_equal), float(target[3] == "S3"), float(not name1 or not name2),
        float(not address1 or not address2), (name_mean + address_mean) / 2.0,
        name_mean * address_mean,
    ]
