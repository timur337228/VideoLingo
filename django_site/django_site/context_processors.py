from django.conf import settings


NOINDEX_PATH_PREFIXES = (
    "/admin/",
    "/auth/",
    "/billing/",
    "/social-auth/",
    "/translate/",
)


def _get_base_url(request):
    configured_base_url = settings.APP_BASE_URL.strip()
    if configured_base_url:
        return configured_base_url
    return request.build_absolute_uri("/").rstrip("/")


def legal_context(request):
    canonical_path = request.path if request.path == "/" else request.path.rstrip("/") + "/"
    app_base_url = _get_base_url(request)
    seo_robots = (
        "noindex, nofollow"
        if request.path.startswith(NOINDEX_PATH_PREFIXES)
        else "index, follow"
    )

    return {
        "app_base_url": app_base_url,
        "legal_service_name": settings.LEGAL_SERVICE_NAME,
        "legal_owner_name": settings.LEGAL_OWNER_NAME,
        "legal_status_label": settings.LEGAL_STATUS_LABEL,
        "legal_inn": settings.LEGAL_INN,
        "legal_contact_email": settings.LEGAL_CONTACT_EMAIL,
        "legal_contact_phone": settings.LEGAL_CONTACT_PHONE,
        "legal_contact_address": settings.LEGAL_CONTACT_ADDRESS,
        "legal_refund_email": settings.LEGAL_REFUND_EMAIL,
        "legal_docs_effective_date": settings.LEGAL_DOCS_EFFECTIVE_DATE,
        "google_auth_enabled": settings.GOOGLE_AUTH_ENABLED,
        "seo_title": settings.LEGAL_SERVICE_NAME or "MixxTranslate",
        "seo_default_description": (
            "MixxTranslate помогает переводить видео, оформлять заказы онлайн "
            "и получать готовый результат в одном интерфейсе."
        ),
        "seo_canonical_url": f"{app_base_url}{canonical_path}",
        "seo_robots": seo_robots,
    }
