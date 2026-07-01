from django.conf import settings
from django.http import Http404
from django.shortcuts import render

from billing.models import BillingSettings, PaymentPackage
from billing.utils import get_package_discount_percent, get_package_rate_per_minute
from .languages import (
    get_featured_language_pairs,
    get_language_by_slug,
    get_languages,
    get_related_language_pairs,
)


def _app_base_url(request):
    configured_base_url = settings.APP_BASE_URL.strip()
    if configured_base_url:
        return configured_base_url
    return request.build_absolute_uri("/").rstrip("/")


def _seo_context(*, title: str, description: str, robots="index, follow"):
    return {
        "seo_title": title,
        "seo_description": description,
        "seo_robots_override": robots,
    }


def _legal_context():
    return {
        "legal_service_name": settings.LEGAL_SERVICE_NAME,
        "legal_owner_name": settings.LEGAL_OWNER_NAME,
        "legal_status_label": settings.LEGAL_STATUS_LABEL,
        "legal_inn": settings.LEGAL_INN,
        "legal_contact_email": settings.LEGAL_CONTACT_EMAIL,
        "legal_contact_phone": settings.LEGAL_CONTACT_PHONE,
        "legal_contact_address": settings.LEGAL_CONTACT_ADDRESS,
        "legal_refund_email": settings.LEGAL_REFUND_EMAIL,
        "legal_docs_effective_date": settings.LEGAL_DOCS_EFFECTIVE_DATE,
        "app_base_url": settings.APP_BASE_URL,
    }


def _public_pricing_context():
    settings_obj = BillingSettings.load()
    packages = list(PaymentPackage.objects.filter(is_active=True))

    for package in packages:
        package.display_rate_per_minute = get_package_rate_per_minute(
            package,
            price_rub_per_minute=settings_obj.price_rub_per_minute,
        )
        package.display_discount_percent = get_package_discount_percent(
            package,
            price_rub_per_minute=settings_obj.price_rub_per_minute,
        )

    return {
        "pricing_packages": packages,
        "base_rate_per_minute": settings_obj.price_rub_per_minute,
        "public_packages": packages[:3],
    }


def _build_pricing_summary(base_rate_per_minute, packages):
    discount_packages = [
        package
        for package in packages
        if getattr(package, "display_discount_percent", 0) > 0
    ]
    if not discount_packages:
        return (
            f"Базовая цена — {base_rate_per_minute:.2f} ₽ за минуту видео. "
            "Стоимость фиксируется до оплаты."
        )

    min_minutes = min(package.minutes for package in discount_packages)
    return (
        f"Базовая цена — {base_rate_per_minute:.2f} ₽ за минуту видео. "
        f"При покупке пакетов от {min_minutes} минут скидка применяется автоматически."
    )


def _build_faq_items(base_rate_per_minute, packages):
    return [
        {
            "question": "Сколько времени занимает перевод видео?",
            "answer": (
                "Обычно перевод занимает от 3 до 15 минут в зависимости от "
                "длительности файла, текущей очереди и сложности исходной дорожки. "
                "В редких случаях обработка может занять больше времени."
            ),
        },
        {
            "question": "Какие форматы видео поддерживаются?",
            "answer": (
                "Сервис принимает MP4, MOV, AVI, MKV, WEBM и M4V. "
                "MP4 — самый распространённый формат, поэтому он подходит лучше всего."
            ),
        },
        {
            "question": "На какие языки можно перевести видео?",
            "answer": (
                "Сейчас доступно 8 языков: английский, русский, французский, "
                "немецкий, итальянский, испанский, японский и китайский."
            ),
        },
        {
            "question": "Сколько стоит перевод видео?",
            "answer": _build_pricing_summary(base_rate_per_minute, packages),
        },
        {
            "question": "Как происходит оплата?",
            "answer": "Оплатить перевод можно банковской картой или через СБП прямо на сайте.",
        },
        {
            "question": "Нужно ли регистрироваться?",
            "answer": (
                "Да. Аккаунт нужен для загрузки файла, отслеживания статуса перевода, "
                "получения результата и хранения истории оплат."
            ),
        },
        {
            "question": "Сохраняется ли качество видео после перевода?",
            "answer": (
                "Да, сервис сохраняет исходное качество видео, а перевод затрагивает "
                "прежде всего звуковую дорожку и дополнительные материалы."
            ),
        },
    ]


def preview(request):
    context = _public_pricing_context()
    context["available_languages"] = get_languages()
    context["featured_language_pairs"] = get_featured_language_pairs()
    context["faq_items"] = _build_faq_items(
        context["base_rate_per_minute"],
        context["pricing_packages"],
    )
    context.update(
        _seo_context(
            title="MixxTranslate — переводчик видео онлайн",
            description=(
                "Онлайн-сервис перевода видео с оплатой картой или через СБП. "
                "Загрузите файл, выберите язык и получите готовый перевод."
            ),
        )
    )
    return render(request, "accounts/preview.html", context)


def offer(request):
    context = _legal_context()
    context.update(
        _seo_context(
            title=f"Публичная оферта | {settings.LEGAL_SERVICE_NAME}",
            description=(
                "Публичная оферта сервиса MixxTranslate: условия оказания услуг "
                "по переводу видео, оплаты, возвратов и использования сайта."
            ),
        )
    )
    return render(request, "legal/offer.html", context)


def privacy_policy(request):
    context = _legal_context()
    context.update(
        _seo_context(
            title=f"Политика обработки персональных данных | {settings.LEGAL_SERVICE_NAME}",
            description=(
                "Политика обработки персональных данных MixxTranslate: какие данные "
                "собираются, как они используются и как защищаются."
            ),
        )
    )
    return render(request, "legal/privacy_policy.html", context)


def contacts(request):
    context = _legal_context()
    context.update(
        _seo_context(
            title=f"Контакты и реквизиты | {settings.LEGAL_SERVICE_NAME}",
            description=(
                "Контакты, реквизиты и каналы связи MixxTranslate для вопросов по "
                "оплате, возвратам, претензиям и обслуживанию."
            ),
        )
    )
    return render(request, "legal/contacts.html", context)


def payments_refunds(request):
    context = _legal_context()
    context.update(
        _seo_context(
            title=f"Оплата и возвраты | {settings.LEGAL_SERVICE_NAME}",
            description=(
                "Условия оплаты и возвратов MixxTranslate: пополнение минут, "
                "зачисление платежей, спорные списания и обращения по возвратам."
            ),
        )
    )
    return render(request, "legal/payments_refunds.html", context)


def language_pair(request, source_slug, target_slug):
    source_language = get_language_by_slug(source_slug)
    target_language = get_language_by_slug(target_slug)

    if not source_language or not target_language or source_language["code"] == target_language["code"]:
        raise Http404("Language pair not found")

    context = _public_pricing_context()
    context["source_language"] = source_language
    context["target_language"] = target_language
    context["available_languages"] = get_languages()
    context["related_language_pairs"] = get_related_language_pairs(
        source_language,
        target_language,
        limit=8,
    )
    context["faq_items"] = _build_faq_items(
        context["base_rate_per_minute"],
        context["pricing_packages"],
    )
    context["pair_heading"] = (
        f"Перевод видео с {source_language['source_case']} на {target_language['target_case']}"
    )
    context["pair_description_text"] = (
        f"MixxTranslate помогает перевести видео с {source_language['source_case']} "
        f"на {target_language['target_case']} онлайн: загрузите файл, выберите язык "
        "и получите готовое видео в личном кабинете."
    )
    context["pair_pricing_text"] = _build_pricing_summary(
        context["base_rate_per_minute"],
        context["pricing_packages"],
    )
    context.update(
        _seo_context(
            title=(
                f"Перевести видео с {source_language['source_case']} "
                f"на {target_language['target_case']} онлайн | MixxTranslate"
            ),
            description=(
                f"Переведите видео с {source_language['source_case']} на "
                f"{target_language['target_case']} онлайн. Загрузите файл, "
                "выберите язык и получите готовое видео с переводом."
            ),
        )
    )
    return render(request, "accounts/language_pair.html", context)


def robots_txt(request):
    response = render(
        request,
        "robots.txt",
        {"sitemap_url": f"{_app_base_url(request)}/sitemap.xml"},
        content_type="text/plain; charset=utf-8",
    )
    return response
