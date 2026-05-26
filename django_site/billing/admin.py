from django.contrib import admin

from .models import BalanceTransaction, BillingSettings, Payment, PaymentPackage


@admin.register(BillingSettings)
class BillingSettingsAdmin(admin.ModelAdmin):
    list_display = ("price_rub_per_minute", "updated_at")


@admin.register(PaymentPackage)
class PaymentPackageAdmin(admin.ModelAdmin):
    list_display = ("name", "minutes", "price_rub_override", "is_active", "sort_order")
    list_filter = ("is_active",)
    list_editable = ("price_rub_override", "is_active", "sort_order")
    ordering = ("sort_order", "minutes")


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "status", "amount_rub", "seconds_to_credit", "provider", "created_at", "paid_at")
    list_filter = ("status", "provider")
    search_fields = ("user__email", "provider_payment_id")
    readonly_fields = ("created_at", "updated_at", "paid_at")


@admin.register(BalanceTransaction)
class BalanceTransactionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "type", "seconds_delta", "payment", "video", "created_at")
    list_filter = ("type",)
    search_fields = ("user__email", "comment")
    readonly_fields = ("created_at",)
