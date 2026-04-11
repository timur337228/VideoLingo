import re
import syllables
from pypinyin import Style, pinyin
sec_per_unit = {
    "en": 0.162, "ru": 0.188,
    "fr": 0.139, "de": 0.168,
    "it": 0.143, "es": 0.128,
    "ja": 0.128, "zh": 0.193,
}

min_speed = {
    "en": 0.90, "ru": 0.90, "fr": 0.90, "de": 0.90,
    "it": 0.90, "es": 0.90, "ja": 0.95, "zh": 0.95,
}

max_speed = {
    "en": 1.28, "ru": 1.22, "fr": 1.35, "de": 1.22,
    "it": 1.35, "es": 1.38, "ja": 1.30, "zh": 1.20,
}

k_lang_voice = {
    "en": 1.0, "ru": 1.0, "fr": 1.0, "de": 1.0,
    "it": 1.0, "es": 1.0, "ja": 1.0, "zh": 1.0,
}


def clamp(x, min_value, max_value):
    return max(min_value, min(x, max_value))


def _count_vowel_groups(text, vowels):
    words = re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)
    total = 0
    vowel_pattern = f"[{re.escape(vowels)}]+"
    for word in words:
        total += max(1, len(re.findall(vowel_pattern, word)))
    return total


def count_units(text, lang):
    if not isinstance(text, str) or not text.strip():
        return 0

    lang = lang.lower()

    if lang == "en":
        total = 0
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        for word in words:
            estimated = syllables.estimate(word)
            if estimated > 0:
                total += estimated
            else:
                total += max(1, len(re.findall(r"[aeiouy]+", word.lower())))
        return max(1, total)

    if lang == "ru":
        return _count_vowel_groups(text, "аеёиоуыэюя")

    if lang == "fr":
        text = re.sub(r"\b([^\W\d_]+?)(?:e|es|ent)\b", r"\1", text.lower(), flags=re.UNICODE)
        return _count_vowel_groups(text, "aeiouyàâéèêëîïôùûüÿœæ")

    if lang == "de":
        return _count_vowel_groups(text, "aeiouyäöü")

    if lang == "it":
        return _count_vowel_groups(text, "aeiouàèéìíîòóùú")

    if lang == "es":
        return _count_vowel_groups(text, "aeiouáéíóúü")

    if lang == "ja":
        normalized = re.sub(
            r"[きぎしじちぢにひびぴみりキギシジチヂニヒビピミリ]"
            r"[ゃゅょャュョ]",
            "X",
            text,
        )
        normalized = re.sub(r"[っッー]", "", normalized)
        return len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]", normalized))

    if lang == "zh":
        han_text = re.sub(r"[^\u4e00-\u9fff]", "", text)
        return len(pinyin(han_text, style=Style.NORMAL))

    words = re.findall(r"\S+", text)
    return len(words)


def punctuation_pause(text, lang):
    if not isinstance(text, str) or not text.strip():
        return 0.0

    comma_like = len(re.findall(r"[,;:，；：、]", text))
    sentence_end = len(re.findall(r"[.!?…。！？]+", text))
    dash_pause = len(re.findall(r"\s[-–—]\s", text))
    number_tokens = len(re.findall(r"\d+(?:[.,:/-]\d+)*", text))
    acronym_tokens = len(re.findall(r"\b(?:[A-ZА-ЯЁ]{2,}|(?:[A-ZА-ЯЁ]\.){2,})\b", text))

    pause = (
        comma_like * 0.10
        + sentence_end * 0.18
        + dash_pause * 0.08
        + number_tokens * 0.08
        + acronym_tokens * 0.06
    )

    if lang.lower() in {"ja", "zh"}:
        pause += len(re.findall(r"\s+", text)) * 0.03

    return round(pause, 3)


def get_coefficient_tts(text, lang, budget_sec):
    lang = lang.lower()
    unit_count = count_units(text, lang)
    pause_duration = punctuation_pause(text, lang)
    d_nat = unit_count * sec_per_unit[lang] + pause_duration
    safe_budget = max(float(budget_sec) - 0.12, 0.25)

    talking_speed = clamp(
        (k_lang_voice[lang] * d_nat / safe_budget) ** 0.9,
        min_speed[lang],
        max_speed[lang],
    )
    return round(talking_speed, 3)
