from django.db import models
from decimal import Decimal
from django.conf import settings


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


class BalanceTransaction(models.Model):
    TYPE_CHOICES = [
        ("topup", "Пополнение"),
        ("reserve", "Резерв"),
        ("confirm", "Подтверждение"),
        ("refund", "Возврат"),
        ("adjustment", "Ручная корректировка")
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="balance_transactions")
    type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    seconds_delta = models.IntegerField()
    rub_amount_snapshot = models.decimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    comment = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)