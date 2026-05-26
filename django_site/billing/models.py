from decimal import Decimal

from django.conf import settings
from django.db import models
from django.utils import timezone


class BillingSettings(models.Model):
    price_rub_per_minute = models.DecimalField(max_digits=8, decimal_places=2, default=Decimal("25.00"))
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    def __str__(self):
        return f"BillingSettings ({self.price_rub_per_minute} RUB/min)"


class PaymentPackage(models.Model):
    name = models.CharField(max_length=120)
    minutes = models.PositiveIntegerField()
    price_rub_override = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    description = models.CharField(max_length=255, blank=True)
    is_active = models.BooleanField(default=True)
    sort_order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ("sort_order", "minutes", "id")

    def __str__(self):
        return self.name

    @property
    def seconds_to_credit(self):
        return self.minutes * 60

    def get_price_rub(self):
        if self.price_rub_override is not None:
            return self.price_rub_override
        settings_obj = BillingSettings.load()
        return (Decimal(self.minutes) * settings_obj.price_rub_per_minute).quantize(Decimal("0.01"))


class Payment(models.Model):
    STATUS_CREATED = "created"
    STATUS_PENDING = "pending"
    STATUS_PAID = "paid"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"
    STATUS_CHOICES = [
        (STATUS_CREATED, "Создан"),
        (STATUS_PENDING, "Ожидает оплаты"),
        (STATUS_PAID, "Оплачен"),
        (STATUS_FAILED, "Ошибка"),
        (STATUS_CANCELLED, "Отменен"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="payments")
    package = models.ForeignKey("billing.PaymentPackage", on_delete=models.SET_NULL, null=True, blank=True, related_name="payments")
    provider = models.CharField(max_length=40, default="stub")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_CREATED)
    amount_rub = models.DecimalField(max_digits=10, decimal_places=2)
    seconds_to_credit = models.PositiveIntegerField()
    description = models.CharField(max_length=255, blank=True)
    provider_payment_id = models.CharField(max_length=255, blank=True)
    provider_payload = models.JSONField(default=dict, blank=True)
    paid_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"Payment #{self.pk} for {self.user.email}"

    def mark_pending(self):
        self.status = self.STATUS_PENDING
        self.save(update_fields=["status", "updated_at"])

    def mark_cancelled(self):
        self.status = self.STATUS_CANCELLED
        self.save(update_fields=["status", "updated_at"])

    def mark_failed(self):
        self.status = self.STATUS_FAILED
        self.save(update_fields=["status", "updated_at"])

    def mark_paid(self):
        self.status = self.STATUS_PAID
        self.paid_at = timezone.now()
        self.save(update_fields=["status", "paid_at", "updated_at"])


class BalanceTransaction(models.Model):
    TYPE_CHOICES = [
        ("topup", "Пополнение"),
        ("reserve", "Резерв"),
        ("confirm", "Подтверждение"),
        ("refund", "Возврат"),
        ("adjustment", "Ручная корректировка"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="balance_transactions")
    payment = models.ForeignKey("billing.Payment", on_delete=models.SET_NULL, null=True, blank=True, related_name="transactions")
    video = models.ForeignKey("translater.Video", on_delete=models.SET_NULL, null=True, blank=True, related_name="balance_transactions")
    type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    seconds_delta = models.IntegerField()
    rub_amount_snapshot = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    comment = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.user.email}: {self.type} ({self.seconds_delta}s)"
