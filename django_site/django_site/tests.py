from django.test import TestCase, override_settings
from django.urls import reverse


@override_settings(
    LEGAL_OWNER_NAME="Иван Иванов",
    LEGAL_INN="123456789012",
    LEGAL_CONTACT_EMAIL="legal@example.com",
)
class LegalPagesTests(TestCase):
    def test_offer_page_renders(self):
        response = self.client.get(reverse("offer"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Публичная оферта")
        self.assertContains(response, "Иван Иванов")

    def test_privacy_policy_page_renders(self):
        response = self.client.get(reverse("privacy_policy"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Политика обработки")
        self.assertContains(response, "legal@example.com")

    def test_contacts_page_renders(self):
        response = self.client.get(reverse("contacts"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Контакты")
        self.assertContains(response, "123456789012")

    def test_payments_refunds_page_renders(self):
        response = self.client.get(reverse("payments_refunds"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Оплата")
        self.assertContains(response, "ЮKassa")

    def test_home_footer_contains_legal_links(self):
        response = self.client.get(reverse("home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("offer"))
        self.assertContains(response, reverse("payments_refunds"))
        self.assertContains(response, reverse("privacy_policy"))
        self.assertContains(response, reverse("contacts"))

    def test_home_shows_public_pricing(self):
        response = self.client.get(reverse("home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Базовая цена")
