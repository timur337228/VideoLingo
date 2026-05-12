from django.urls import path

from .views import (
    complete_registration,
    login_view,
    logout_view,
    register,
    verify_email,
    verify_email_sent,
)

urlpatterns = [
    path("register/", register, name="register"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("verify-email-sent/", verify_email_sent, name="verify_email_sent"),
    path("verify-email/<uuid:token>/", verify_email, name="verify_email"),
    path("complete-registration/", complete_registration, name="complete_registration"),
]
