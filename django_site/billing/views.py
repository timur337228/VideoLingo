import json
from decimal import Decimal, InvalidOperation

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .models import BillingSettings, Payment, PaymentPackage
from .utils import (
    build_checkout_context,
    calculate_custom_topup,
    create_custom_payment,
    create_payment,
    get_package_discount_percent,
    get_package_rate_per_minute,
    get_provider_confirmation_url,
    start_yookassa_payment,
    sync_payment_with_provider,
)
from .yookassa import ALLOWED_PAYMENT_METHODS, YooKassaAPIError, YooKassaConfigurationError


def _build_dashboard_context(request, *, custom_amount_value="", custom_payment_error=""):
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

    payments = request.user.payments.all()[:10]
    transactions = request.user.balance_transactions.all()[:20]
    custom_topup_preview = None

    if custom_amount_value:
        try:
            preview_amount = Decimal(custom_amount_value.replace(",", ".")).quantize(Decimal("0.01"))
            if preview_amount > 0:
                custom_topup_preview = calculate_custom_topup(preview_amount, packages=packages)
        except (InvalidOperation, AttributeError):
            custom_topup_preview = None

    return {
        "billing_settings": settings_obj,
        "packages": packages,
        "payments": payments,
        "transactions": transactions,
        "custom_amount_value": custom_amount_value,
        "custom_payment_error": custom_payment_error,
        "custom_topup_preview": custom_topup_preview,
    }


@login_required
def billing_dashboard(request):
    return render(
        request,
        "billing/dashboard.html",
        _build_dashboard_context(request),
    )


@login_required
@require_POST
def create_payment_view(request, package_id):
    package = get_object_or_404(PaymentPackage, pk=package_id, is_active=True)
    payment = create_payment(request.user, package)
    return redirect("billing_checkout", payment_id=payment.pk)


@login_required
@require_POST
def create_custom_payment_view(request):
    raw_amount = (request.POST.get("amount_rub") or "").strip()
    normalized_amount = raw_amount.replace(",", ".")

    try:
        amount_rub = Decimal(normalized_amount).quantize(Decimal("0.01"))
    except InvalidOperation:
        context = _build_dashboard_context(
            request,
            custom_amount_value=raw_amount,
            custom_payment_error="Введите корректную сумму в рублях.",
        )
        return render(request, "billing/dashboard.html", context, status=400)

    if amount_rub <= 0:
        context = _build_dashboard_context(
            request,
            custom_amount_value=raw_amount,
            custom_payment_error="Сумма должна быть больше нуля.",
        )
        return render(request, "billing/dashboard.html", context, status=400)

    try:
        payment = create_custom_payment(request.user, amount_rub)
    except ValueError:
        context = _build_dashboard_context(
            request,
            custom_amount_value=raw_amount,
            custom_payment_error="Не удалось подготовить оплату. Проверьте сумму и попробуйте ещё раз.",
        )
        return render(request, "billing/dashboard.html", context, status=400)

    return redirect("billing_checkout", payment_id=payment.pk)


@login_required
def payment_checkout(request, payment_id):
    payment = get_object_or_404(Payment, pk=payment_id, user=request.user)
    payment = sync_payment_with_provider(payment)
    checkout_context = build_checkout_context(payment, request)
    return render(request, "billing/checkout.html", checkout_context)


@login_required
@require_POST
def start_payment_view(request, payment_id):
    payment = get_object_or_404(Payment, pk=payment_id, user=request.user)
    method_type = (request.POST.get("method_type") or "").strip()

    if method_type not in ALLOWED_PAYMENT_METHODS:
        return HttpResponseBadRequest("Unsupported payment method")

    try:
        payment = start_yookassa_payment(payment, request, method_type)
    except YooKassaConfigurationError as exc:
        context = build_checkout_context(payment, request)
        context["payment_error"] = str(exc)
        return render(request, "billing/checkout.html", context, status=503)
    except YooKassaAPIError as exc:
        context = build_checkout_context(payment, request)
        context["payment_error"] = (
            "Не удалось создать платеж в YooKassa. Попробуйте еще раз. "
            f"Техническая причина: {exc}"
        )
        return render(request, "billing/checkout.html", context, status=502)

    confirmation_url = get_provider_confirmation_url(payment)
    if payment.status == Payment.STATUS_PAID:
        return redirect("billing_success", payment_id=payment.pk)
    if not confirmation_url:
        context = build_checkout_context(payment, request)
        context["payment_error"] = "YooKassa не вернула ссылку на оплату."
        return render(request, "billing/checkout.html", context, status=502)
    return redirect(confirmation_url)


@login_required
def payment_success(request, payment_id):
    payment = get_object_or_404(Payment, pk=payment_id, user=request.user)
    payment = sync_payment_with_provider(payment)
    if payment.status == Payment.STATUS_CANCELLED:
        return redirect("billing_cancel", payment_id=payment.pk)
    return render(request, "billing/success.html", {"payment": payment})


@login_required
def payment_cancel(request, payment_id):
    payment = get_object_or_404(Payment, pk=payment_id, user=request.user)
    payment = sync_payment_with_provider(payment)
    if payment.status == Payment.STATUS_PAID:
        return redirect("billing_success", payment_id=payment.pk)
    if payment.provider != "yookassa" and payment.status not in {Payment.STATUS_PAID, Payment.STATUS_FAILED}:
        payment.mark_cancelled()
    return render(request, "billing/cancel.html", {"payment": payment})


@csrf_exempt
@require_POST
def payment_webhook(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return HttpResponseBadRequest("Invalid JSON payload")

    payment_object = payload.get("object") or {}
    provider_payment_id = payment_object.get("id")
    if not provider_payment_id:
        return HttpResponseBadRequest("object.id is required")

    payment = Payment.objects.filter(provider_payment_id=provider_payment_id).first()
    if payment is None:
        metadata = payment_object.get("metadata") or {}
        payment_id = metadata.get("payment_id")
        if payment_id:
            payment = get_object_or_404(Payment, pk=payment_id)
        else:
            return HttpResponseBadRequest("Payment not found")

    try:
        payment = sync_payment_with_provider(payment, raise_errors=True)
    except (YooKassaAPIError, YooKassaConfigurationError):
        return JsonResponse({"ok": False, "error": "Unable to verify payment in YooKassa"}, status=502)

    return JsonResponse({"ok": True, "payment_status": payment.status})
