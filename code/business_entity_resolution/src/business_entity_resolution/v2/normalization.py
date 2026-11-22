import re

from ..normalize import normalize_address, normalize_country, normalize_name


_EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.%+-]+@[\w.-]+\.[a-z]{2,}(?![\w.-])")
_PHONE_LABEL_RE = re.compile(
    r"(?i)\b(?:phone|telephone|tel|mobile|mob|ph)\s*[:#.]?\s*"
    r"(\+?\d[\d\s()./-]{5,}\d)"
)
_PHONE_PLUS_RE = re.compile(r"(?<!\w)\+\d[\d\s()./-]{5,}\d(?!\w)")
_PHONE_GROUPED_RE = re.compile(
    r"(?<!\w)(?:\(\d{2,4}\)\s*)?\d{2,4}(?:[-./]\s*\d{2,4}){2,4}(?!\w)"
)
_US_POSTAL_RE = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?\s*$")
_INDIA_POSTAL_RE = re.compile(r"(?<!\d)([1-8]\d{5})\s*$")
_FR_POSTAL_RE = re.compile(
    r"(?<!\d)(\d{5})(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'-]*)?\s*$"
)

_SUFFIX_COLLAPSES = (
    (("professional", "limited", "liability", "co"), ("pllc",)),
    (("limited", "liability", "partnership"), ("llp",)),
    (("limited", "liability", "co"), ("llc",)),
    (("public", "limited", "co"), ("plc",)),
    (("s", "a", "s"), ("sas",)),
    (("s", "a", "r", "l"), ("sarl",)),
    (("g", "m", "b", "h"), ("gmbh",)),
    (("b", "v"), ("bv",)),
    (("p", "t", "e"), ("pte",)),
    (("p", "t", "y"), ("pty",)),
    (("s", "a"), ("sa",)),
)
_LEGAL_SUFFIXES = {
    "ag", "bv", "co", "corp", "gmbh", "inc", "llc", "llp", "ltd", "plc",
    "pllc", "pte", "pty", "pvt", "sa", "sarl", "sas",
}
_ADDRESS_DIRECTIONS = {
    "n": "north", "s": "south", "e": "east", "w": "west",
}


def normalize_name_v2(value):
    """Keep the baseline cleanup and canonicalize only recognized terminal legal forms."""
    parts = normalize_name(value).split()
    for suffix, replacement in _SUFFIX_COLLAPSES:
        if len(parts) >= len(suffix) and tuple(parts[-len(suffix):]) == suffix:
            parts = parts[:-len(suffix)] + list(replacement)
            break
    return " ".join(parts)


def normalize_address_v2(value):
    """Reuse baseline address cleanup and expand only standalone compass abbreviations."""
    parts = normalize_address(value).split()
    return " ".join(_ADDRESS_DIRECTIONS.get(part, part) for part in parts)


def strip_legal_suffix_v2(value):
    parts = normalize_name_v2(value).split()
    while parts and parts[-1] in _LEGAL_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _canonical_phone(raw, country):
    digits = "".join(char for char in raw if char.isdigit())
    if not 7 <= len(digits) <= 15:
        return ""
    country = normalize_country(country)
    if country in {"us", "usa", "united states", "united states of america"}:
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
    elif country == "india":
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        elif len(digits) == 11 and digits.startswith("0"):
            digits = digits[1:]
    elif country == "france" and len(digits) == 11 and digits.startswith("33"):
        digits = "0" + digits[2:]
    return digits


def _postal_codes(address, country):
    country = normalize_country(country)
    address = (address or "").strip().rstrip(" ,.;")
    if country in {"us", "usa", "united states", "united states of america"}:
        match = _US_POSTAL_RE.search(address)
    elif country == "india":
        match = _INDIA_POSTAL_RE.search(address)
    elif country == "france":
        match = _FR_POSTAL_RE.search(address)
    else:
        match = None
    return (match.group(1),) if match else ()


def extract_identifiers_v2(business_name, business_address, country):
    """Extract only well-formed values embedded in text fields; absent columns stay absent."""
    name = business_name or ""
    address = business_address or ""
    text = f"{name} {address}"
    emails = {match.group(0).casefold().rstrip(".,;:") for match in _EMAIL_RE.finditer(text)}

    phone_values = []
    phone_values.extend(match.group(1) for match in _PHONE_LABEL_RE.finditer(text))
    phone_values.extend(match.group(0) for match in _PHONE_PLUS_RE.finditer(text))
    phone_values.extend(match.group(0) for match in _PHONE_GROUPED_RE.finditer(text))
    phones = {phone for raw in phone_values if (phone := _canonical_phone(raw, country))}

    postals = set(_postal_codes(address, country))
    return {
        "emails": tuple(sorted(emails)),
        "phones": tuple(sorted(phones)),
        "postals": tuple(sorted(postals)),
    }
