import math
from decimal import Decimal, ROUND_DOWN

from django.db import transaction
from django.urls import reverse
from django.utils import timezone

from .models import BalanceTransaction, BillingSettings, Payment, PaymentPackage


class InsufficientBalanceError(ValueError):
    pass


def seconds_to_charge(duration_seconds: int) -> int:
    safe_duration = max(int(duration_seconds or 0), 1)
    return math.ceil(safe_duration / 60) * 60


def rub_cost(duration_seconds: int, price_rub_per_minute: Decimal) -> Decimal:
    charged_seconds = seconds_to_charge(duration_seconds)
    minutes = Decimal(charged_seconds) / Decimal("60")
    return (minutes * price_rub_per_minute).quantize(Decimal("0.01"))


def get_price_for_seconds(duration_seconds: int) -> Decimal:
    settings_obj = BillingSettings.load()
    return rub_cost(duration_seconds, settings_obj.price_rub_per_minute)


def get_package_rate_per_minute(package: PaymentPackage, price_rub_per_minute: Decimal | None = None) -> Decimal:
    base_rate = price_rub_per_minute or BillingSettings.load().price_rub_per_minute
    package_minutes = Decimal(package.minutes)
    if package_minutes <= 0:
        return base_rate
    return (package.get_price_rub() / package_minutes).quantize(Decimal("0.0001"))


def get_package_discount_percent(package: PaymentPackage, price_rub_per_minute: Decimal | None = None) -> Decimal:
    base_rate = price_rub_per_minute or BillingSettings.load().price_rub_per_minute
    if base_rate <= 0:
        return Decimal("0.00")

    rate_per_minute = get_package_rate_per_minute(package, price_rub_per_minute=base_rate)
    discount = (Decimal("1") - (rate_per_minute / base_rate)) * Decimal("100")
    if discount <= 0:
        return Decimal("0.00")
    return discount.quantize(Decimal("0.01"))


def calculate_custom_topup(amount_rub: Decimal, packages=None):
    settings_obj = BillingSettings.load()
    base_rate = settings_obj.price_rub_per_minute

    best_result = {
        "amount_rub": amount_rub.quantize(Decimal("0.01")),
        "seconds_to_credit": 0,
        "credited_minutes": Decimal("0.00"),
        "applied_package": None,
        "rate_per_minute": base_rate,
        "discount_percent": Decimal("0.00"),
    }

    def build_candidate(rate_per_minute: Decimal, package: PaymentPackage | None):
        if rate_per_minute <= 0:
            return None

        credited_seconds = int(
            ((amount_rub / rate_per_minute) * Decimal("60")).to_integral_value(rounding=ROUND_DOWN)
        )
        if credited_seconds <= 0:
            return None
        if package is not None and credited_seconds < package.seconds_to_credit:
            return None

        if package is None:
            discount_percent = Decimal("0.00")
        else:
            discount_percent = get_package_discount_percent(package, price_rub_per_minute=base_rate)

        return {
            "amount_rub": amount_rub.quantize(Decimal("0.01")),
            "seconds_to_credit": credited_seconds,
            "credited_minutes": (Decimal(credited_seconds) / Decimal("60")).quantize(Decimal("0.01")),
            "applied_package": package,
            "rate_per_minute": rate_per_minute,
            "discount_percent": discount_percent,
        }

    candidates = [build_candidate(base_rate, None)]
    package_qs = packages
    if package_qs is None:
        package_qs = PaymentPackage.objects.filter(is_active=True)

    for package in package_qs:
        candidates.append(
            build_candidate(
                get_package_rate_per_minute(package, price_rub_per_minute=base_rate),
                package,
            )
        )

    valid_candidates = [candidate for candidate in candidates if candidate is not None]
    if not valid_candidates:
        return best_result

    return max(
        valid_candidates,
        key=lambda candidate: (
            candidate["seconds_to_credit"],
            candidate["discount_percent"],
            candidate["applied_package"].minutes if candidate["applied_package"] else 0,
        ),
    )


def reserve_video_minutes(user, duration_seconds, video):
    charged_seconds = seconds_to_charge(duration_seconds)
    charged_rub = get_price_for_seconds(duration_seconds)

    with transaction.atomic():
        user.refresh_from_db()
        if user.available_seconds < charged_seconds:
            raise InsufficientBalanceError("Недостаточно доступных минут для запуска перевода.")

        user.available_seconds -= charged_seconds
        user.save(update_fields=["available_seconds"])

        video.duration_seconds = duration_seconds
        video.charged_seconds = charged_seconds
        video.is_charge_confirmed = False
        video.is_refunded = False
        video.save(update_fields=["duration_seconds", "charged_seconds", "is_charge_confirmed", "is_refunded"])

        BalanceTransaction.objects.create(
            user=user,
            video=video,
            type="reserve",
            seconds_delta=-charged_seconds,
            rub_amount_snapshot=charged_rub,
            comment=f"Резерв минут под видео #{video.pk}",
        )

    return charged_seconds


def confirm_video_charge(video):
    if video.is_charge_confirmed or video.charged_seconds <= 0:
        return False

    with transaction.atomic():
        video.refresh_from_db()
        if video.is_charge_confirmed or video.charged_seconds <= 0:
            return False

        BalanceTransaction.objects.create(
            user=video.user,
            video=video,
            type="confirm",
            seconds_delta=0,
            rub_amount_snapshot=get_price_for_seconds(video.duration_seconds),
            comment=f"Подтверждение списания за видео #{video.pk}",
        )
        video.is_charge_confirmed = True
        video.save(update_fields=["is_charge_confirmed"])

    return True


def refund_video_charge(video, comment="Возврат минут за неуспешную обработку"):
    if video.is_refunded or video.charged_seconds <= 0:
        return False

    with transaction.atomic():
        video.refresh_from_db()
        if video.is_refunded or video.charged_seconds <= 0:
            return False

        user = video.user
        user.refresh_from_db()
        user.available_seconds += video.charged_seconds
        user.save(update_fields=["available_seconds"])

        BalanceTransaction.objects.create(
            user=user,
            video=video,
            type="refund",
            seconds_delta=video.charged_seconds,
            rub_amount_snapshot=get_price_for_seconds(video.duration_seconds),
            comment=comment,
        )

        video.is_refunded = True
        video.save(update_fields=["is_refunded"])

    return True


def create_payment(user, package: PaymentPackage, provider="stub"):
    amount_rub = package.get_price_rub()
    payment = Payment.objects.create(
        user=user,
        package=package,
        provider=provider,
        status=Payment.STATUS_PENDING,
        amount_rub=amount_rub,
        seconds_to_credit=package.seconds_to_credit,
        description=f"Пополнение баланса: {package.minutes} мин.",
    )
    return payment


def create_custom_payment(user, amount_rub: Decimal, provider="stub"):
    custom_topup = calculate_custom_topup(amount_rub)
    seconds_to_credit = custom_topup["seconds_to_credit"]
    applied_package = custom_topup["applied_package"]
    credited_minutes = custom_topup["credited_minutes"]

    if seconds_to_credit < 60:
        raise ValueError("Сумма слишком маленькая: нужно оплатить хотя бы 1 минуту.")

    description = f"Пополнение баланса: {credited_minutes.quantize(Decimal('0.01'))} мин."
    if applied_package is not None and custom_topup["discount_percent"] > 0:
        description += (
            f" Применена скидка {custom_topup['discount_percent']}% "
            f"по пакету «{applied_package.name}»."
        )

    payment = Payment.objects.create(
        user=user,
        provider=provider,
        status=Payment.STATUS_PENDING,
        amount_rub=custom_topup["amount_rub"],
        seconds_to_credit=seconds_to_credit,
        description=description,
    )
    return payment


def build_checkout_context(payment: Payment, request):
    return {
        "payment": payment,
        "gateway_payload": {
            "payment_id": payment.pk,
            "amount_rub": str(payment.amount_rub),
            "description": payment.description,
            "seconds_to_credit": payment.seconds_to_credit,
            "success_url": request.build_absolute_uri(reverse("billing_success", kwargs={"payment_id": payment.pk})),
            "cancel_url": request.build_absolute_uri(reverse("billing_cancel", kwargs={"payment_id": payment.pk})),
            "webhook_url": request.build_absolute_uri(reverse("billing_webhook")),
            "metadata": {
                "payment_id": payment.pk,
                "user_id": payment.user_id,
            },
        },
    }


def mark_payment_paid(payment: Payment, provider_payment_id="", payload=None):
    if payment.status == Payment.STATUS_PAID:
        return payment

    with transaction.atomic():
        payment.refresh_from_db()
        if payment.status == Payment.STATUS_PAID:
            return payment

        user = payment.user
        user.refresh_from_db()
        user.available_seconds += payment.seconds_to_credit
        user.save(update_fields=["available_seconds"])

        BalanceTransaction.objects.create(
            user=user,
            payment=payment,
            type="topup",
            seconds_delta=payment.seconds_to_credit,
            rub_amount_snapshot=payment.amount_rub,
            comment=f"Оплата #{payment.pk}",
        )

        payment.provider_payment_id = provider_payment_id or payment.provider_payment_id
        payment.provider_payload = payload or payment.provider_payload
        payment.status = Payment.STATUS_PAID
        payment.paid_at = timezone.now()
        payment.save(
            update_fields=[
                "provider_payment_id",
                "provider_payload",
                "status",
                "paid_at",
                "updated_at",
            ]
        )

    return payment


def mark_payment_status(payment: Payment, status: str, provider_payment_id="", payload=None):
    if status == Payment.STATUS_PAID:
        return mark_payment_paid(payment, provider_payment_id=provider_payment_id, payload=payload)

    payment.provider_payment_id = provider_payment_id or payment.provider_payment_id
    payment.provider_payload = payload or payment.provider_payload

    if status == Payment.STATUS_FAILED:
        payment.status = Payment.STATUS_FAILED
    elif status == Payment.STATUS_CANCELLED:
        payment.status = Payment.STATUS_CANCELLED
    else:
        payment.status = Payment.STATUS_PENDING

    payment.save(
        update_fields=[
            "provider_payment_id",
            "provider_payload",
            "status",
            "updated_at",
        ]
    )

    return payment
