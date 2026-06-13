from django.contrib.auth.tokens import default_token_generator
from django.core import mail
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode

from .models import PendingRegistration, User


@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class PasswordResetFlowTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(
            email="user@example.com",
            password="OldPassword123!",
            is_active=True,
        )
        self.google_user = User.objects.create_user(
            email="google@example.com",
            password="UnusedPassword123!",
            is_active=True,
            is_google_auth=True,
        )

    def build_reset_url(self, user, token=None):
        uidb64 = urlsafe_base64_encode(force_bytes(user.pk))
        token = token or default_token_generator.make_token(user)
        return reverse(
            "complite_reset_password",
            kwargs={"uidb64": uidb64, "token": token},
        )

    def test_reset_password_page_shows_login_button(self):
        response = self.client.get(reverse("reset_password"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Вернуться ко входу")

    def test_reset_password_sends_email_for_regular_user(self):
        response = self.client.post(
            reverse("reset_password"),
            {"email": self.user.email},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].subject, "Сброс пароля")

        uidb64 = urlsafe_base64_encode(force_bytes(self.user.pk))
        self.assertIn(uidb64, mail.outbox[0].body)
        self.assertIn("/auth/complite-reset-password/", mail.outbox[0].body)

    def test_reset_password_hides_google_user_existence(self):
        response = self.client.post(
            reverse("reset_password"),
            {"email": self.google_user.email},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Если аккаунт существует и для него доступен вход по паролю",
        )
        self.assertEqual(len(mail.outbox), 0)

    def test_reset_password_hides_missing_user_existence(self):
        response = self.client.post(
            reverse("reset_password"),
            {"email": "missing@example.com"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Если аккаунт существует и для него доступен вход по паролю",
        )
        self.assertEqual(len(mail.outbox), 0)

    def test_reset_confirm_page_opens_with_valid_token(self):
        response = self.client.get(self.build_reset_url(self.user))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Сохранить новый пароль")

    def test_reset_confirm_changes_password(self):
        new_password = "NewStrongPassword123!"
        response = self.client.post(
            self.build_reset_url(self.user),
            {
                "new_password1": new_password,
                "new_password2": new_password,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Пароль успешно")

        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password(new_password))

    def test_reset_confirm_shows_invalid_page_for_bad_token(self):
        response = self.client.get(
            self.build_reset_url(self.user, token="invalid-token")
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ссылка для сброса")


class AuthSecurityTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(
            email="user@example.com",
            password="StrongPassword123!",
            is_active=True,
        )

    def test_logout_requires_post(self):
        self.client.force_login(self.user)

        response = self.client.get(reverse("logout"))

        self.assertEqual(response.status_code, 405)

    def test_logout_post_logs_user_out(self):
        self.client.force_login(self.user)

        response = self.client.post(reverse("logout"))

        self.assertRedirects(response, reverse("home"))
        self.assertNotIn("_auth_user_id", self.client.session)

    @override_settings(LOGIN_RATE_LIMIT_ATTEMPTS=3, LOGIN_RATE_LIMIT_WINDOW_SECONDS=60)
    def test_login_is_rate_limited_after_repeated_failures(self):
        for _ in range(3):
            self.client.post(
                reverse("login"),
                {"username": self.user.email, "password": "wrong-password"},
            )

        response = self.client.post(
            reverse("login"),
            {"username": self.user.email, "password": "wrong-password"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Слишком много попыток входа. Попробуйте позже.")


@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class RegistrationLegalTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_register_requires_legal_acceptance(self):
        response = self.client.post(
            reverse("register"),
            {"email": "new@example.com"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Нужно принять оферту")
        self.assertFalse(PendingRegistration.objects.filter(email="new@example.com").exists())

    def test_register_stores_legal_acceptance_metadata(self):
        response = self.client.post(
            reverse("register"),
            {"email": "new@example.com", "accept_legal": "on"},
        )

        self.assertRedirects(response, reverse("verify_email_sent"))
        pending = PendingRegistration.objects.get(email="new@example.com")
        self.assertIsNotNone(pending.offer_accepted_at)
        self.assertIsNotNone(pending.privacy_policy_accepted_at)
        self.assertTrue(pending.legal_docs_version)
