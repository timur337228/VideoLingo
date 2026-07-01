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
        self.assertContains(response, "Частые вопросы")
        self.assertContains(response, "С английского на русский")

    @override_settings(APP_BASE_URL="https://mixxtranslate.ru")
    def test_robots_txt_exposes_sitemap_and_private_disallows(self):
        response = self.client.get(reverse("robots_txt"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/plain; charset=utf-8")
        self.assertContains(response, "User-agent: *")
        self.assertContains(response, "Allow: /")
        self.assertContains(response, "Disallow: /billing/")
        self.assertContains(response, "Sitemap: https://mixxtranslate.ru/sitemap.xml")

    @override_settings(
        APP_BASE_URL="https://mixxtranslate.ru",
        ALLOWED_HOSTS=["mixxtranslate.ru", "testserver", "localhost", "127.0.0.1"],
    )
    def test_sitemap_lists_public_pages(self):
        response = self.client.get(
            reverse("django.contrib.sitemaps.views.sitemap"),
            HTTP_HOST="mixxtranslate.ru",
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "https://mixxtranslate.ru/")
        self.assertContains(response, "https://mixxtranslate.ru/offer/")
        self.assertContains(response, "https://mixxtranslate.ru/privacy-policy/")
        self.assertContains(response, "https://mixxtranslate.ru/contacts/")
        self.assertContains(response, "https://mixxtranslate.ru/payments-refunds/")
        self.assertContains(response, "https://mixxtranslate.ru/translate-video/english-to-russian/")

    @override_settings(APP_BASE_URL="https://mixxtranslate.ru")
    def test_home_has_canonical_and_meta_description(self):
        response = self.client.get(reverse("home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<link rel="canonical" href="https://mixxtranslate.ru/">', html=True)
        self.assertContains(
            response,
            'name="description" content="Переводите видео онлайн бесплатно. Загрузите файл, выберите язык — получите готовое видео с переводом."',
            html=False,
        )
        self.assertContains(
            response,
            'name="keywords" content="переводчик видео, перевести видео онлайн, видео перевод, mixxtranslate"',
            html=False,
        )
        self.assertContains(response, 'content="index, follow"', html=False)
        self.assertContains(response, "Переводчик видео онлайн — загрузите видео, выберите язык и получите перевод за минуты.")
        self.assertContains(response, 'rel="icon" href="/static/favicon.ico"', html=False)

    def test_login_page_is_noindex(self):
        response = self.client.get(reverse("login"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="robots" content="noindex, nofollow"', html=False)

    def test_google_verification_file_is_served(self):
        response = self.client.get("/google62cd21c736d230fd.html")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "google-site-verification: google62cd21c736d230fd.html")

    def test_yandex_verification_file_is_served(self):
        response = self.client.get("/yandex_7cd360ccdc45f960.html")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Verification: 7cd360ccdc45f960")

    def test_language_pair_page_renders(self):
        response = self.client.get(
            reverse(
                "language_pair",
                kwargs={"source_slug": "english", "target_slug": "russian"},
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Перевод видео с английского на русский")
        self.assertContains(response, "Загрузите файл")
