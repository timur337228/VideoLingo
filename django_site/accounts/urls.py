from django.urls import path

from .views import (
    complete_registration,
    login_view,
    logout_view,
    register,
    verify_email,
    verify_email_sent,
    reset_password,
    complite_reset_password,
)

urlpatterns = [
    path("register/", register, name="register"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("verify-email-sent/", verify_email_sent, name="verify_email_sent"),
    path("verify-email/<uuid:token>/", verify_email, name="verify_email"),
    path("complete-registration/", complete_registration, name="complete_registration"),
    path("reset-password/", reset_password, name="reset_password"),
    path("complite-reset-password/<uidb64>/<token>/", complite_reset_password, name="complite_reset_password"),
]
