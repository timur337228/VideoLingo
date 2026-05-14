from django.shortcuts import resolve_url

from social_core.exceptions import AuthCanceled, AuthConnectionError
from social_django.middleware import SocialAuthExceptionMiddleware


class CustomSocialAuthExceptionMiddleware(SocialAuthExceptionMiddleware):
    def get_message(self, request, exception):
        if isinstance(exception, AuthCanceled):
            return "Вход через Google отменён. Попробуйте выбрать аккаунт ещё раз."
        if isinstance(exception, AuthConnectionError):
            return "Не удалось связаться с Google. Проверьте интернет на сервере и попробуйте ещё раз."
        return "Не удалось выполнить вход через Google. Попробуйте ещё раз."

    def get_redirect_uri(self, request, exception):
        strategy = getattr(request, "social_strategy", None)
        if strategy is not None:
            next_url = strategy.session_get("next")
            if next_url:
                return next_url
        return resolve_url("login")
