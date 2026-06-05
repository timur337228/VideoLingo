import uuid

import requests
from django.conf import settings


API_BASE_URL = "https://api.yookassa.ru/v3"
ALLOWED_PAYMENT_METHODS = frozenset({"bank_card", "sbp"})


class YooKassaError(Exception):
    pass


class YooKassaConfigurationError(YooKassaError):
    pass


class YooKassaAPIError(YooKassaError):
    pass


def is_configured() -> bool:
    return bool(settings.YOOKASSA_SHOP_ID and settings.YOOKASSA_SECRET_KEY)


def _auth():
    if not is_configured():
        raise YooKassaConfigurationError(
            "YooKassa не настроена: добавьте YOOKASSA_SHOP_ID и YOOKASSA_SECRET_KEY в .env."
        )
    return settings.YOOKASSA_SHOP_ID, settings.YOOKASSA_SECRET_KEY


def _request(method: str, path: str, *, payload=None, idempotence_key: str | None = None):
    headers = {
        "Accept": "application/json",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if idempotence_key:
        headers["Idempotence-Key"] = idempotence_key

    response = requests.request(
        method=method,
        url=f"{API_BASE_URL}{path}",
        auth=_auth(),
        headers=headers,
        json=payload,
        timeout=30,
    )

    try:
        data = response.json()
    except ValueError:
        data = {}

    if response.status_code >= 400:
        description = data.get("description") or data.get("message") or response.text
        raise YooKassaAPIError(
            f"YooKassa API error {response.status_code}: {description or 'unknown error'}"
        )

    return data


def create_payment(*, payment, method_type: str, return_url: str):
    if method_type not in ALLOWED_PAYMENT_METHODS:
        raise YooKassaAPIError("Неподдерживаемый способ оплаты.")

    payload = {
        "amount": {
            "value": str(payment.amount_rub),
            "currency": "RUB",
        },
        "capture": True,
        "payment_method_data": {
            "type": method_type,
        },
        "confirmation": {
            "type": "redirect",
            "return_url": return_url,
        },
        "description": payment.description[:128],
        "metadata": {
            "payment_id": str(payment.pk),
            "user_id": str(payment.user_id),
        },
    }

    return _request(
        "POST",
        "/payments",
        payload=payload,
        idempotence_key=str(uuid.uuid4()),
    )


def get_payment(provider_payment_id: str):
    return _request("GET", f"/payments/{provider_payment_id}")
