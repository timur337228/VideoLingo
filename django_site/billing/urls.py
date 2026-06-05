from django.urls import path

from .views import (
    billing_dashboard,
    create_custom_payment_view,
    create_payment_view,
    payment_cancel,
    payment_checkout,
    payment_success,
    start_payment_view,
    payment_webhook,
)

urlpatterns = [
    path("", billing_dashboard, name="billing_dashboard"),
    path("packages/<int:package_id>/create-payment/", create_payment_view, name="billing_create_payment"),
    path("payments/custom/create/", create_custom_payment_view, name="billing_create_custom_payment"),
    path("payments/<int:payment_id>/checkout/", payment_checkout, name="billing_checkout"),
    path("payments/<int:payment_id>/start/", start_payment_view, name="billing_start_payment"),
    path("payments/<int:payment_id>/success/", payment_success, name="billing_success"),
    path("payments/<int:payment_id>/cancel/", payment_cancel, name="billing_cancel"),
    path("webhook/", payment_webhook, name="billing_webhook"),
]
