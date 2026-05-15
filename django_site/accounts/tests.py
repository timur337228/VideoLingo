from django.contrib.auth.tokens import default_token_generator
from django.core import mail
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode

from .models import User


@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class PasswordResetFlowTests(TestCase):
    def setUp(self):
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

    def test_reset_password_rejects_google_user(self):
        response = self.client.post(
            reverse("reset_password"),
            {"email": self.google_user.email},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Вы авторизовались через google, войдите с помощью google",
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
