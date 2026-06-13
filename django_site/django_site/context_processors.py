from django.conf import settings


def legal_context(request):
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
    }
