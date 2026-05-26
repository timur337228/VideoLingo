from decimal import Decimal

from django.test import TestCase
from django.urls import reverse

from accounts.models import User
from billing.models import BalanceTransaction, BillingSettings, PaymentPackage
from billing.utils import (
    calculate_custom_topup,
    create_custom_payment,
    create_payment,
    mark_payment_paid,
    refund_video_charge,
    reserve_video_minutes,
)
from translater.models import Video


class BillingFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="billing@example.com",
            password="StrongPassword123!",
            is_active=True,
            available_seconds=600,
        )
        self.video = Video.objects.create(
            user=self.user,
            task_id="test-task-id",
        )
        BillingSettings.load()

    def test_reserve_video_minutes_decreases_balance_and_updates_video(self):
        reserve_video_minutes(self.user, 61, self.video)

        self.user.refresh_from_db()
        self.video.refresh_from_db()

        self.assertEqual(self.user.available_seconds, 480)
        self.assertEqual(self.video.charged_seconds, 120)
        self.assertFalse(self.video.is_refunded)
        self.assertEqual(
            BalanceTransaction.objects.filter(video=self.video, type="reserve").count(),
            1,
        )

    def test_refund_video_charge_returns_reserved_seconds(self):
        reserve_video_minutes(self.user, 61, self.video)

        refunded = refund_video_charge(self.video)

        self.assertTrue(refunded)
        self.user.refresh_from_db()
        self.video.refresh_from_db()
        self.assertEqual(self.user.available_seconds, 600)
        self.assertTrue(self.video.is_refunded)

    def test_mark_payment_paid_adds_balance_once(self):
        package = PaymentPackage.objects.create(
            name="Starter",
            minutes=10,
            price_rub_override="250.00",
        )
        payment = create_payment(self.user, package)

        mark_payment_paid(payment, provider_payment_id="provider-123")
        mark_payment_paid(payment, provider_payment_id="provider-123")

        self.user.refresh_from_db()
        payment.refresh_from_db()

        self.assertEqual(self.user.available_seconds, 1200)
        self.assertEqual(payment.status, payment.STATUS_PAID)
        self.assertEqual(
            BalanceTransaction.objects.filter(payment=payment, type="topup").count(),
            1,
        )

    def test_calculate_custom_topup_applies_best_package_discount(self):
        BillingSettings.objects.filter(pk=1).update(price_rub_per_minute=Decimal("25.00"))
        starter = PaymentPackage.objects.create(
            name="30 минут",
            minutes=30,
            price_rub_override=Decimal("712.50"),
        )
        PaymentPackage.objects.create(
            name="60 минут",
            minutes=60,
            price_rub_override=Decimal("1350.00"),
        )

        topup = calculate_custom_topup(Decimal("736.25"))

        self.assertEqual(topup["seconds_to_credit"], 1860)
        self.assertEqual(topup["credited_minutes"], Decimal("31.00"))
        self.assertEqual(topup["applied_package"], starter)
        self.assertEqual(topup["discount_percent"], Decimal("5.00"))

    def test_create_custom_payment_creates_discounted_topup(self):
        BillingSettings.objects.filter(pk=1).update(price_rub_per_minute=Decimal("25.00"))
        PaymentPackage.objects.create(
            name="30 минут",
            minutes=30,
            price_rub_override=Decimal("712.50"),
        )

        payment = create_custom_payment(self.user, Decimal("736.25"))

        self.assertIsNone(payment.package)
        self.assertEqual(payment.amount_rub, Decimal("736.25"))
        self.assertEqual(payment.seconds_to_credit, 1860)
        self.assertIn("5.00%", payment.description)

    def test_create_custom_payment_view_redirects_to_checkout(self):
        BillingSettings.objects.filter(pk=1).update(price_rub_per_minute=Decimal("25.00"))
        PaymentPackage.objects.create(
            name="30 минут",
            minutes=30,
            price_rub_override=Decimal("712.50"),
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("billing_create_custom_payment"),
            {"amount_rub": "736.25"},
        )

        payment = self.user.payments.latest("id")
        self.assertRedirects(response, reverse("billing_checkout", kwargs={"payment_id": payment.pk}))
        self.assertEqual(payment.seconds_to_credit, 1860)
