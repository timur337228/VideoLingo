from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "core" / "tts_backend" / "inworld_created_voices.json"
DESIGN_URL = "https://api.inworld.ai/voices/v1/voices:design"
PUBLISH_URL_TEMPLATE = "https://api.inworld.ai/voices/v1/voices/{voice_id}:publish"
TARGET_VOICES_PER_GENDER = 7
TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
MAX_REQUEST_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 45
MIN_PREVIEW_TEXT_LENGTHS = {
    "default": 70,
    "ja": 55,
    "zh": 45,
}
EXISTING_VOICE_COUNTS: dict[str, dict[str, int]] = {
    "ru": {"male": 2, "female": 2},
    "de": {"male": 1, "female": 1},
    "zh": {"male": 2, "female": 1},
    "fr": {"male": 3, "female": 1},
    "es": {"male": 5, "female": 3},
    "ja": {"male": 2, "female": 2},
}
GENERATION_PLAN: dict[str, dict[str, int]] = {
    language: {
        "male": max(0, TARGET_VOICES_PER_GENDER - counts.get("male", 0)),
        "female": max(0, TARGET_VOICES_PER_GENDER - counts.get("female", 0)),
    }
    for language, counts in EXISTING_VOICE_COUNTS.items()
}


LANGUAGE_SETTINGS: dict[str, dict[str, Any]] = {
    "en": {
        "lang_code": "EN_US",
        "language_name": "English",
        "accent": "neutral American English",
        "preview_texts": [
            "Hello, I am ready to help you translate and voice your video naturally.",
            "Let us make this narration clear, warm, and easy to follow from start to finish.",
            "I can guide the listener smoothly, with calm timing and clean pronunciation.",
            "This voice should feel natural, polished, and comfortable for long listening sessions.",
            "Every sentence needs to sound expressive without becoming theatrical or exaggerated.",
            "Keep the delivery confident, human, and easy to understand at a steady pace.",
            "The final voice should work well for narration, tutorials, and translated videos.",
            "I speak with clarity, stable rhythm, and a balanced tone that feels trustworthy.",
        ],
    },
    "ru": {
        "lang_code": "RU_RU",
        "language_name": "Russian",
        "accent": "neutral Moscow Russian",
        "preview_texts": [
            "Здравствуйте, я готов помочь озвучить и перевести ваше видео естественным голосом.",
            "Пусть эта речь звучит спокойно, понятно и приятно для долгого прослушивания.",
            "Важно сохранить ровный темп, чистую дикцию и живую человеческую подачу.",
            "Такой голос должен хорошо подходить для озвучки роликов, гайдов и объяснений.",
            "Каждая фраза должна звучать выразительно, но без лишней театральности.",
            "Нужна уверенная, тёплая и естественная манера речи с хорошей разборчивостью.",
            "Этот голос должен вести слушателя плавно и понятно от начала до конца.",
            "Хочется получить качественную, мягкую и профессиональную русскую озвучку.",
        ],
    },
    "fr": {
        "lang_code": "FR_FR",
        "language_name": "French",
        "accent": "neutral Parisian French",
        "preview_texts": [
            "Bonjour, je suis pret a donner a votre video une voix claire, naturelle et fluide.",
            "Cette voix doit rester chaleureuse, precise et agreable a ecouter du debut a la fin.",
            "Je parle avec un rythme regulier, une diction nette et une presence rassurante.",
            "L'objectif est une narration elegante, humaine et facile a suivre.",
            "Chaque phrase doit sonner vivante sans devenir trop theatrale ou trop rapide.",
            "Cette voix convient bien aux explications, tutoriels et videos traduites.",
            "Je veux une interpretation stable, expressive et professionnelle en francais.",
            "Le rendu final doit etre naturel, confiant et confortable pour une longue ecoute.",
        ],
    },
    "de": {
        "lang_code": "DE_DE",
        "language_name": "German",
        "accent": "neutral High German",
        "preview_texts": [
            "Hallo, ich bin bereit, Ihr Video mit einer klaren und natuerlichen Stimme zu vertonen.",
            "Diese Stimme soll ruhig, praezise und ueber lange Zeit angenehm anzuhoeren sein.",
            "Ich spreche mit sauberer Aussprache, gutem Rhythmus und sicherer Praesenz.",
            "Das Ergebnis soll professionell, menschlich und leicht verstaendlich wirken.",
            "Jeder Satz braucht Ausdruck, aber ohne kuenstlich oder uebertrieben zu klingen.",
            "Diese Stimme passt gut zu Erklaervideos, Tutorials und uebersetzten Inhalten.",
            "Wichtig sind ein gleichmaessiges Tempo, Vertrauen und natuerliche Betonung.",
            "Die Wiedergabe soll warm, stabil und klar vom Anfang bis zum Ende sein.",
        ],
    },
    "it": {
        "lang_code": "IT_IT",
        "language_name": "Italian",
        "accent": "standard Italian with a neutral Northern Italian accent",
        "preview_texts": [
            "Ciao, sono pronto a dare al tuo video una voce naturale, chiara e piacevole.",
            "Questa voce deve risultare calda, precisa e facile da seguire fino alla fine.",
            "Parlo con ritmo regolare, buona dizione e una presenza umana rassicurante.",
            "L'obiettivo e una narrazione fluida, professionale e adatta a contenuti tradotti.",
            "Ogni frase deve suonare espressiva senza diventare teatrale o troppo rapida.",
            "Questa voce si adatta bene a tutorial, spiegazioni e video informativi.",
            "Voglio una resa stabile, naturale e confidente in italiano.",
            "Il risultato finale deve essere pulito, credibile e comodo da ascoltare a lungo.",
        ],
    },
    "es": {
        "lang_code": "ES_ES",
        "language_name": "Spanish",
        "accent": "neutral Castilian Spanish",
        "preview_texts": [
            "Hola, estoy listo para dar a tu video una voz natural, clara y agradable.",
            "Esta voz debe sonar cercana, precisa y facil de escuchar de principio a fin.",
            "Hablo con ritmo estable, buena diccion y una presencia humana confiable.",
            "Buscamos una narracion fluida, profesional y comoda para videos traducidos.",
            "Cada frase debe tener expresion sin sonar exagerada ni demasiado teatral.",
            "Esta voz funciona bien para tutoriales, explicaciones y contenido educativo.",
            "Quiero una interpretacion clara, segura y natural en espanol.",
            "El resultado final debe sentirse calido, equilibrado y facil de seguir.",
        ],
    },
    "ja": {
        "lang_code": "JA_JP",
        "language_name": "Japanese",
        "accent": "standard Tokyo Japanese",
        "preview_texts": [
            "こんにちは。自然で聞き取りやすい声で、動画の翻訳音声を仕上げます。",
            "この声は、最初から最後まで落ち着いて心地よく聞こえる必要があります。",
            "はっきりした発音と安定したテンポで、自然な話し方を目指します。",
            "説明動画やチュートリアルに合う、信頼感のあるナレーションが理想です。",
            "大げさになりすぎず、ほどよい表現力のある読み上げにしたいです。",
            "長く聞いても疲れにくい、滑らかで整った日本語の声が必要です。",
            "この音声は、温かさと明瞭さを両立した自然な仕上がりを目指します。",
            "聞き手を迷わせない、安定感のあるプロ品質の声にしたいです。",
        ],
    },
    "zh": {
        "lang_code": "ZH_CN",
        "language_name": "Chinese",
        "accent": "standard Mandarin with a neutral Beijing accent",
        "preview_texts": [
            "你好，我准备好用自然清晰的声音为你的视频完成配音。",
            "这个声音需要从头到尾都保持稳定、温和并且容易理解。",
            "我会用清楚的咬字、自然的节奏和可靠的表达来进行旁白。",
            "这个声音应该适合教程、讲解和翻译后的视频内容。",
            "每一句话都要有表现力，但不能夸张或显得做作。",
            "希望最终效果听起来真实、专业，而且适合长时间收听。",
            "这段配音需要兼顾温暖感、清晰度和稳定的语气。",
            "理想的声音应该自然顺畅，让听众轻松跟上内容节奏。",
        ],
    },
}


MALE_STYLES = [
    {
        "distinctive": "Confident, grounded",
        "age": "30-40 years old",
        "tone": "warm and dependable",
        "delivery": "clear conversational delivery",
        "pacing": "steady medium pacing",
        "extra": "natural low-mid timbre with clean articulation",
    },
    {
        "distinctive": "Calm, thoughtful",
        "age": "35-45 years old",
        "tone": "measured and reassuring",
        "delivery": "smooth explanatory delivery",
        "pacing": "slightly slow, deliberate pacing",
        "extra": "subtle depth and a relaxed vocal texture",
    },
    {
        "distinctive": "Energetic, articulate",
        "age": "mid-20s to early 30s",
        "tone": "bright and engaged",
        "delivery": "crisp presenter-style delivery",
        "pacing": "lively but controlled pacing",
        "extra": "clean projection with natural enthusiasm",
    },
    {
        "distinctive": "Rich, cinematic",
        "age": "40-50 years old",
        "tone": "serious but welcoming",
        "delivery": "polished narration style",
        "pacing": "steady pacing with natural pauses",
        "extra": "slightly resonant timbre without sounding overly dramatic",
    },
    {
        "distinctive": "Friendly, modern",
        "age": "late 20s to mid-30s",
        "tone": "open and upbeat",
        "delivery": "natural digital-host delivery",
        "pacing": "balanced pacing",
        "extra": "smooth, youthful texture with precise consonants",
    },
    {
        "distinctive": "Soft-spoken, intelligent",
        "age": "30-45 years old",
        "tone": "calm and professional",
        "delivery": "gentle instructional delivery",
        "pacing": "slower, easy-to-follow pacing",
        "extra": "subtle breath warmth and refined clarity",
    },
    {
        "distinctive": "Authoritative, composed",
        "age": "45-55 years old",
        "tone": "stable and trustworthy",
        "delivery": "broadcast-style delivery",
        "pacing": "measured pacing with clean emphasis",
        "extra": "firm low register with polished diction",
    },
    {
        "distinctive": "Expressive, warm",
        "age": "30-40 years old",
        "tone": "human and inviting",
        "delivery": "storytelling-friendly delivery",
        "pacing": "smooth medium pacing",
        "extra": "natural texture with subtle emotional color",
    },
]


FEMALE_STYLES = [
    {
        "distinctive": "Warm, soothing",
        "age": "28-38 years old",
        "tone": "gentle and reassuring",
        "delivery": "smooth conversational delivery",
        "pacing": "steady medium pacing",
        "extra": "soft timbre with clear articulation",
    },
    {
        "distinctive": "Bright, articulate",
        "age": "mid-20s to early 30s",
        "tone": "friendly and confident",
        "delivery": "clean presenter-style delivery",
        "pacing": "lively but controlled pacing",
        "extra": "light, polished texture with natural sparkle",
    },
    {
        "distinctive": "Calm, intelligent",
        "age": "30-40 years old",
        "tone": "professional and composed",
        "delivery": "clear instructional delivery",
        "pacing": "slightly slow, deliberate pacing",
        "extra": "balanced mid-range tone with subtle warmth",
    },
    {
        "distinctive": "Elegant, mature",
        "age": "40-50 years old",
        "tone": "poised and trustworthy",
        "delivery": "polished narration style",
        "pacing": "steady pacing with natural pauses",
        "extra": "refined texture and smooth phrasing",
    },
    {
        "distinctive": "Energetic, modern",
        "age": "late 20s to mid-30s",
        "tone": "upbeat and engaging",
        "delivery": "natural digital-host delivery",
        "pacing": "balanced pacing",
        "extra": "fresh vocal color with crisp consonants",
    },
    {
        "distinctive": "Soft, empathetic",
        "age": "30-45 years old",
        "tone": "kind and comforting",
        "delivery": "gentle explanatory delivery",
        "pacing": "slow, easy-to-follow pacing",
        "extra": "airy warmth with stable clarity",
    },
    {
        "distinctive": "Confident, broadcast-ready",
        "age": "35-45 years old",
        "tone": "clear and dependable",
        "delivery": "broadcast-style delivery",
        "pacing": "measured pacing with clean emphasis",
        "extra": "focused mid-low register and polished diction",
    },
    {
        "distinctive": "Expressive, inviting",
        "age": "28-38 years old",
        "tone": "human and approachable",
        "delivery": "storytelling-friendly delivery",
        "pacing": "smooth medium pacing",
        "extra": "natural emotional color with a soft finish",
    },
]


DEFAULT_DATA = {
    "meta": {
        "created_at": None,
        "updated_at": None,
        "request_interval_seconds": None,
        "publish_enabled": True,
    },
    "voices": {},
    "pending_publish": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slowly design and publish Inworld voices with a fixed delay between every request."
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(GENERATION_PLAN.keys()),
        choices=list(GENERATION_PLAN.keys()),
        help="Language groups to generate from the built-in missing-voices plan.",
    )
    parser.add_argument(
        "--genders",
        nargs="+",
        default=["male", "female"],
        choices=["male", "female"],
        help="Gender groups to generate.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Seconds to wait between every HTTP request.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="numberOfSamples sent to voices:design.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save created voice IDs.",
    )
    parser.add_argument(
        "--max-voices",
        type=int,
        default=None,
        help="Stop after this many successfully completed voices in the current run.",
    )
    parser.add_argument(
        "--design-only",
        action="store_true",
        help="Only design preview voices and do not publish them.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Start with a fresh JSON file instead of resuming.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("INWORLD_TTS_KEY")
    if not api_key:
        raise RuntimeError("INWORLD_TTS_KEY not found in .env")
    return api_key


def load_data(output_path: Path, overwrite: bool) -> dict[str, Any]:
    if overwrite or not output_path.exists():
        data = deepcopy(DEFAULT_DATA)
        data["meta"]["created_at"] = now_iso()
        return data

    with output_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_data(output_path: Path, data: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data["meta"]["updated_at"] = now_iso()
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def ensure_language_bucket(data: dict[str, Any], short_code: str) -> dict[str, Any]:
    settings = LANGUAGE_SETTINGS[short_code]
    voices = data.setdefault("voices", {})
    bucket = voices.setdefault(
        short_code,
        {
            "language_name": settings["language_name"],
            "lang_code": settings["lang_code"],
            "male": [],
            "female": [],
        },
    )
    bucket.setdefault("male", [])
    bucket.setdefault("female", [])
    return bucket


def build_design_prompt(short_code: str, gender: str, index: int) -> str:
    settings = LANGUAGE_SETTINGS[short_code]
    style = MALE_STYLES[index] if gender == "male" else FEMALE_STYLES[index]
    gender_text = "male voice" if gender == "male" else "female voice"
    return (
        f"{style['distinctive']} {gender_text}, {settings['language_name']} with a {settings['accent']}, "
        f"{style['age']}. {style['tone'].capitalize()} tone, {style['delivery']}, "
        f"{style['pacing']}. {style['extra'].capitalize()}. Perfect broadcast quality audio."
    )


def build_preview_text(short_code: str, index: int) -> str:
    texts = LANGUAGE_SETTINGS[short_code]["preview_texts"]
    return texts[index]


def get_preview_separator(short_code: str) -> str:
    return "" if short_code in {"ja", "zh"} else " "


def build_preview_text_variants(short_code: str, index: int) -> list[str]:
    texts = LANGUAGE_SETTINGS[short_code]["preview_texts"]
    base_text = texts[index].strip()
    variants = [base_text]
    min_length = MIN_PREVIEW_TEXT_LENGTHS.get(short_code, MIN_PREVIEW_TEXT_LENGTHS["default"])
    if len(base_text) >= min_length:
        return variants

    separator = get_preview_separator(short_code)
    expanded = base_text
    for extra_index in range(index + 1, len(texts)):
        expanded = f"{expanded}{separator}{texts[extra_index].strip()}".strip()
        if expanded not in variants:
            variants.append(expanded)
        if len(expanded) >= min_length:
            break

    if len(variants[-1]) < min_length:
        for extra_index in range(0, index):
            expanded = f"{expanded}{separator}{texts[extra_index].strip()}".strip()
            if expanded not in variants:
                variants.append(expanded)
            if len(expanded) >= min_length:
                break

    return variants


def build_display_name(short_code: str, gender: str, index: int) -> str:
    language_name = LANGUAGE_SETTINGS[short_code]["language_name"]
    return f"VideoLingo {language_name} {gender.capitalize()} {index + 1:02d}"


def get_completed_slots(entries: list[dict[str, Any]]) -> set[int]:
    completed_statuses = {"designed", "published"}
    slots: set[int] = set()
    for entry in entries:
        if entry.get("status") in completed_statuses and isinstance(entry.get("slot"), int):
            slots.add(entry["slot"])
    return slots


def request_json(
    session: requests.Session,
    pacer: "RequestPacer",
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    step_name: str,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = pacer.post(session, url, headers=headers, json=payload, timeout=120)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= MAX_REQUEST_ATTEMPTS:
                break
            backoff = RETRY_BACKOFF_SECONDS * attempt
            print(
                f"[retry] {step_name} network error on attempt {attempt}/{MAX_REQUEST_ATTEMPTS}: {exc}. "
                f"Sleeping {backoff}s before retry",
                flush=True,
            )
            time.sleep(backoff)
            continue

        if response.ok:
            return response.json()

        error_message = f"{step_name} failed with status {response.status_code}: {response.text[:2000]}"
        if response.status_code not in TRANSIENT_STATUS_CODES or attempt >= MAX_REQUEST_ATTEMPTS:
            raise RuntimeError(error_message)

        backoff = RETRY_BACKOFF_SECONDS * attempt
        print(
            f"[retry] {error_message}. Sleeping {backoff}s before retry {attempt + 1}/{MAX_REQUEST_ATTEMPTS}",
            flush=True,
        )
        time.sleep(backoff)

    if last_error is not None:
        raise RuntimeError(f"{step_name} failed after {MAX_REQUEST_ATTEMPTS} attempts: {last_error}") from last_error
    raise RuntimeError(f"{step_name} failed after {MAX_REQUEST_ATTEMPTS} attempts")


class RequestPacer:
    def __init__(self, interval_seconds: float):
        self.interval_seconds = interval_seconds
        self.last_request_started_at: float | None = None

    def post(self, session: requests.Session, url: str, **kwargs: Any) -> requests.Response:
        if self.last_request_started_at is not None:
            elapsed = time.monotonic() - self.last_request_started_at
            remaining = self.interval_seconds - elapsed
            if remaining > 0:
                print(f"[wait] sleeping {remaining:.1f}s before next request", flush=True)
                time.sleep(remaining)
        self.last_request_started_at = time.monotonic()
        return session.post(url, **kwargs)


def design_voice(
    session: requests.Session,
    pacer: RequestPacer,
    headers: dict[str, str],
    short_code: str,
    gender: str,
    index: int,
    samples: int,
) -> tuple[dict[str, Any], str]:
    settings = LANGUAGE_SETTINGS[short_code]
    last_error: Exception | None = None
    for preview_text in build_preview_text_variants(short_code, index):
        payload = {
            "langCode": settings["lang_code"],
            "designPrompt": build_design_prompt(short_code, gender, index),
            "previewText": preview_text,
            "voiceDesignConfig": {"numberOfSamples": samples},
        }
        try:
            response_data = request_json(session, pacer, DESIGN_URL, headers, payload, "Voice design")
        except RuntimeError as exc:
            last_error = exc
            error_text = str(exc).lower()
            if "invalid duration" in error_text:
                print(
                    f"[retry] preview text was rejected for {short_code}/{gender} slot {index + 1}, trying a longer text",
                    flush=True,
                )
                continue
            raise

        preview_voices = response_data.get("previewVoices") or []
        if not preview_voices:
            raise RuntimeError("Voice design succeeded but previewVoices is empty")
        return preview_voices[0], preview_text

    if last_error is not None:
        raise last_error
    raise RuntimeError("Voice design failed before any request was sent")


def publish_voice(
    session: requests.Session,
    pacer: RequestPacer,
    headers: dict[str, str],
    short_code: str,
    gender: str,
    index: int,
    preview_voice_id: str,
) -> dict[str, Any]:
    payload = {
        "displayName": build_display_name(short_code, gender, index),
        "description": build_design_prompt(short_code, gender, index),
        "tags": ["videolingo", short_code, gender, f"slot-{index + 1:02d}"],
    }
    publish_url = PUBLISH_URL_TEMPLATE.format(voice_id=preview_voice_id)
    return request_json(session, pacer, publish_url, headers, payload, "Voice publish")


def append_pending_publish(
    data: dict[str, Any],
    short_code: str,
    gender: str,
    index: int,
    preview_voice: dict[str, Any],
    preview_text: str,
) -> None:
    data.setdefault("pending_publish", []).append(
        {
            "language": short_code,
            "gender": gender,
            "slot": index + 1,
            "preview_voice_id": preview_voice.get("voiceId"),
            "preview_text": preview_text,
            "design_prompt": build_design_prompt(short_code, gender, index),
            "saved_at": now_iso(),
        }
    )


def run() -> int:
    args = parse_args()
    api_key = load_api_key()
    headers = {
        "Authorization": f"Basic {api_key}",
        "Content-Type": "application/json",
    }

    data = load_data(args.output, args.overwrite_output)
    data["meta"]["request_interval_seconds"] = args.interval
    data["meta"]["publish_enabled"] = not args.design_only
    save_data(args.output, data)

    session = requests.Session()
    pacer = RequestPacer(args.interval)
    created_this_run = 0

    try:
        for short_code in args.languages:
            bucket = ensure_language_bucket(data, short_code)
            for gender in args.genders:
                target_count = GENERATION_PLAN[short_code].get(gender, 0)
                if target_count <= 0:
                    print(f"[skip] {short_code}/{gender}: nothing requested in plan", flush=True)
                    continue
                completed_slots = get_completed_slots(bucket[gender])
                if len(completed_slots) >= target_count:
                    print(
                        f"[skip] {short_code}/{gender}: already have {len(completed_slots)} voices",
                        flush=True,
                    )
                    continue

                for slot_number in range(1, target_count + 1):
                    if slot_number in completed_slots:
                        print(
                            f"[skip] {short_code}/{gender}: slot {slot_number} already saved",
                            flush=True,
                        )
                        continue
                    if args.max_voices is not None and created_this_run >= args.max_voices:
                        print("[done] reached --max-voices limit", flush=True)
                        save_data(args.output, data)
                        return 0

                    index = slot_number - 1
                    print(
                        f"[create] {short_code}/{gender} voice {slot_number}/{target_count}",
                        flush=True,
                    )
                    preview_voice, used_preview_text = design_voice(
                        session=session,
                        pacer=pacer,
                        headers=headers,
                        short_code=short_code,
                        gender=gender,
                        index=index,
                        samples=args.samples,
                    )

                    entry = {
                        "display_name": build_display_name(short_code, gender, index),
                        "voice_id": preview_voice.get("voiceId"),
                        "preview_voice_id": preview_voice.get("voiceId"),
                        "lang_code": LANGUAGE_SETTINGS[short_code]["lang_code"],
                        "language_name": LANGUAGE_SETTINGS[short_code]["language_name"],
                        "gender": gender,
                        "slot": slot_number,
                        "preview_text": used_preview_text,
                        "design_prompt": build_design_prompt(short_code, gender, index),
                        "status": "designed",
                    }

                    if not args.design_only:
                        try:
                            published_voice = publish_voice(
                                session=session,
                                pacer=pacer,
                                headers=headers,
                                short_code=short_code,
                                gender=gender,
                                index=index,
                                preview_voice_id=preview_voice["voiceId"],
                            )
                        except Exception:
                            append_pending_publish(
                                data,
                                short_code,
                                gender,
                                index,
                                preview_voice,
                                used_preview_text,
                            )
                            save_data(args.output, data)
                            raise

                        entry["voice_id"] = published_voice.get("voiceId", entry["voice_id"])
                        entry["status"] = "published"
                        entry["published_name"] = published_voice.get("name")
                        entry["tags"] = published_voice.get("tags", [])

                    bucket[gender].append(entry)
                    created_this_run += 1
                    save_data(args.output, data)

    except KeyboardInterrupt:
        print("\n[stop] interrupted by user, progress saved", file=sys.stderr, flush=True)
        save_data(args.output, data)
        return 130

    save_data(args.output, data)
    print("[done] all requested voices have been processed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
