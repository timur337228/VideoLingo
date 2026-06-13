from django.conf import settings
from django.shortcuts import render

from billing.models import BillingSettings, PaymentPackage
from billing.utils import get_package_discount_percent, get_package_rate_per_minute


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
        "base_rate_per_minute": settings_obj.price_rub_per_minute,
        "public_packages": packages[:3],
    }


def preview(request):
    return render(request, "accounts/preview.html", _public_pricing_context())


def offer(request):
    return render(request, "legal/offer.html", _legal_context())


def privacy_policy(request):
    return render(request, "legal/privacy_policy.html", _legal_context())


def contacts(request):
    return render(request, "legal/contacts.html", _legal_context())


def payments_refunds(request):
    return render(request, "legal/payments_refunds.html", _legal_context())
