import re
import unicodedata


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

_LEGAL_ALIASES = {
    "incorporated": "inc", "inc.": "inc", "corp.": "corp", "corporation": "corp",
    "limited": "ltd", "ltd.": "ltd", "private": "pvt", "pvt.": "pvt",
    "company": "co", "co.": "co",
}
_ADDRESS_ALIASES = {
    "rd": "road", "rd.": "road", "st": "street", "st.": "street",
    "str": "street", "ave": "avenue", "ave.": "avenue", "blvd": "boulevard",
    "hwy": "highway", "ln": "lane", "dr": "drive", "ste": "suite",
    "apt": "apartment", "fl": "floor", "pkwy": "parkway",
}


def _unicode_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value or "").casefold()


def normalize_name(value: str) -> str:
    text = _unicode_text(value).replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    tokens = _TOKEN_RE.findall(text)
    tokens = [_LEGAL_ALIASES.get(token, token) for token in tokens]
    return " ".join(tokens)


def normalize_address(value: str) -> str:
    text = _unicode_text(value).replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    tokens = _TOKEN_RE.findall(text)
    tokens = [_ADDRESS_ALIASES.get(token, token) for token in tokens]
    return " ".join(tokens)


def normalize_country(value: str) -> str:
    return _SPACE_RE.sub(" ", _unicode_text(value)).strip()


def tokens(value: str) -> tuple:
    return tuple(_TOKEN_RE.findall(value or ""))

