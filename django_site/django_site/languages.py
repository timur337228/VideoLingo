LANGUAGE_CATALOG = (
    {
        "code": "en",
        "label": "Английский 🇺🇸",
        "name_ru": "Английский",
        "native_name": "English",
        "emoji": "🇺🇸",
        "slug": "english",
        "source_case": "английского",
        "target_case": "английский",
    },
    {
        "code": "ru",
        "label": "Русский 🇷🇺",
        "name_ru": "Русский",
        "native_name": "Русский",
        "emoji": "🇷🇺",
        "slug": "russian",
        "source_case": "русского",
        "target_case": "русский",
    },
    {
        "code": "fr",
        "label": "Французский 🇫🇷",
        "name_ru": "Французский",
        "native_name": "Français",
        "emoji": "🇫🇷",
        "slug": "french",
        "source_case": "французского",
        "target_case": "французский",
    },
    {
        "code": "de",
        "label": "Немецкий 🇩🇪",
        "name_ru": "Немецкий",
        "native_name": "Deutsch",
        "emoji": "🇩🇪",
        "slug": "german",
        "source_case": "немецкого",
        "target_case": "немецкий",
    },
    {
        "code": "it",
        "label": "Итальянский 🇮🇹",
        "name_ru": "Итальянский",
        "native_name": "Italiano",
        "emoji": "🇮🇹",
        "slug": "italian",
        "source_case": "итальянского",
        "target_case": "итальянский",
    },
    {
        "code": "es",
        "label": "Испанский 🇪🇸",
        "name_ru": "Испанский",
        "native_name": "Español",
        "emoji": "🇪🇸",
        "slug": "spanish",
        "source_case": "испанского",
        "target_case": "испанский",
    },
    {
        "code": "ja",
        "label": "Японский 🇯🇵",
        "name_ru": "Японский",
        "native_name": "日本語",
        "emoji": "🇯🇵",
        "slug": "japanese",
        "source_case": "японского",
        "target_case": "японский",
    },
    {
        "code": "zh",
        "label": "Китайский 🇨🇳",
        "name_ru": "Китайский",
        "native_name": "中文",
        "emoji": "🇨🇳",
        "slug": "chinese",
        "source_case": "китайского",
        "target_case": "китайский",
    },
)

LANGUAGE_CHOICES = tuple((item["code"], item["label"]) for item in LANGUAGE_CATALOG)
LANGUAGE_BY_CODE = {item["code"]: item for item in LANGUAGE_CATALOG}
LANGUAGE_BY_SLUG = {item["slug"]: item for item in LANGUAGE_CATALOG}


def get_languages():
    return list(LANGUAGE_CATALOG)


def get_language_by_code(code: str):
    return LANGUAGE_BY_CODE.get(code)


def get_language_by_slug(slug: str):
    return LANGUAGE_BY_SLUG.get(slug)


def build_language_pair(source: dict, target: dict):
    return {
        "source": source,
        "target": target,
        "slug": f"{source['slug']}-to-{target['slug']}",
        "title": f"С {source['source_case']} на {target['target_case']}",
        "route_label": f"{source['emoji']} {source['name_ru']} → {target['emoji']} {target['name_ru']}",
        "cta_label": f"Перевести видео на {target['target_case']}",
    }


def get_language_pairs():
    pairs = []
    for source in LANGUAGE_CATALOG:
        for target in LANGUAGE_CATALOG:
            if source["code"] == target["code"]:
                continue
            pairs.append(build_language_pair(source, target))
    return pairs


def get_featured_language_pairs(limit=8):
    featured = []
    seen = set()
    russian = get_language_by_code("ru")
    english = get_language_by_code("en")

    def add_pair(source, target):
        if not source or not target or source["code"] == target["code"]:
            return
        pair = build_language_pair(source, target)
        if pair["slug"] in seen:
            return
        seen.add(pair["slug"])
        featured.append(pair)

    if english and russian:
        add_pair(english, russian)
        add_pair(russian, english)

    if russian:
        for language in LANGUAGE_CATALOG:
            add_pair(language, russian)
        for language in LANGUAGE_CATALOG:
            add_pair(russian, language)

    for source in LANGUAGE_CATALOG:
        for target in LANGUAGE_CATALOG:
            add_pair(source, target)
            if len(featured) >= limit:
                return featured[:limit]

    return featured[:limit]


def get_related_language_pairs(source: dict, target: dict, limit=8):
    related = []
    seen = set()

    def add_pair(candidate_source, candidate_target):
        if candidate_source["code"] == candidate_target["code"]:
            return
        pair = build_language_pair(candidate_source, candidate_target)
        if pair["slug"] == f"{source['slug']}-to-{target['slug']}":
            return
        if pair["slug"] in seen:
            return
        seen.add(pair["slug"])
        related.append(pair)

    for language in LANGUAGE_CATALOG:
        add_pair(source, language)
        if len(related) >= limit:
            return related[:limit]

    for language in LANGUAGE_CATALOG:
        add_pair(language, target)
        if len(related) >= limit:
            return related[:limit]

    for candidate_source in LANGUAGE_CATALOG:
        for candidate_target in LANGUAGE_CATALOG:
            add_pair(candidate_source, candidate_target)
            if len(related) >= limit:
                return related[:limit]

    return related[:limit]
