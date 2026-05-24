import math
from decimal import Decimal
from django.db import transaction
from .models import BalanceTransaction

def seconds_to_charge(duration_seconds: int) -> int:
    return math.ceil(duration_seconds/60) * 60

def rub_cost(duration_seconds: int, price_rub_per_minute: Decimal) -> Decimal:
    minutes = Decimal(duration_seconds) / Decimal("60")
    return (minutes * price_rub_per_minute).quantize(Decimal("0.01"))



def reserve_video_minutes(user, duration_seconds, video=None):
    charged_seconds = seconds_to_charge(duration_seconds)

    with transaction.atomic():
        user.refresh_from_db()
        if user.available_seconds < charged_seconds:
            raise ValueError("Недостаточно доступных минут")
        
        user.available_seconds -= charged_seconds
        user.save(update_fields=["available_seconds"])

        BalanceTransaction.objects.create(
            user=user,
            type="refund",
            seconds_delta=video.charged_seconds,
            comment=f"Возврат за видео {video.pk}",
        )

        video.is_refunded = True
        video.save(update_fields=["is_refunded"])

