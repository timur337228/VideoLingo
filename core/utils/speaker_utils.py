import math
import re


_EMPTY_VALUES = {"", "nan", "none", "null"}
_GENDER_ALIASES = {
    "m": "male",
    "male": "male",
    "man": "male",
    "masculine": "male",
    "f": "female",
    "female": "female",
    "woman": "female",
    "feminine": "female",
}


def normalize_speaker_id(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value).strip()
    if text.casefold() in _EMPTY_VALUES:
        return None

    # Pandas often reloads integer-like speaker ids as floats (for example 1.0).
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".", 1)[0]

    return text


def normalize_gender_label(value):
    if value is None:
        return None

    text = str(value).strip().casefold()
    return _GENDER_ALIASES.get(text)
