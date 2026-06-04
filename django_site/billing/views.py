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
    mark_payment_status,
)


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
    checkout_context = build_checkout_context(payment, request)
    return render(request, "billing/checkout.html", checkout_context)


@login_required
def payment_success(request, payment_id):
    payment = get_object_or_404(Payment, pk=payment_id, user=request.user)
    return render(request, "billing/success.html", {"payment": payment})


@login_required
def payment_cancel(request, payment_id):
    payment = get_object_or_404(Payment, pk=payment_id, user=request.user)
    if payment.status not in {Payment.STATUS_PAID, Payment.STATUS_FAILED}:
        payment.mark_cancelled()
    return render(request, "billing/cancel.html", {"payment": payment})


@csrf_exempt
@require_POST
def payment_webhook(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return HttpResponseBadRequest("Invalid JSON payload")

    payment_id = payload.get("payment_id")
    status = payload.get("status")
    if not payment_id or not status:
        return HttpResponseBadRequest("payment_id and status are required")

    payment = get_object_or_404(Payment, pk=payment_id)
    mark_payment_status(
        payment,
        status=status,
        provider_payment_id=payload.get("provider_payment_id", ""),
        payload=payload,
    )
    return JsonResponse({"ok": True, "payment_status": payment.status})
