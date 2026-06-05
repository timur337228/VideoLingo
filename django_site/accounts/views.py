import uuid
from datetime import timedelta

from django.conf import settings
from django.contrib.auth import login, logout
from django.contrib.auth.forms import SetPasswordForm
from django.contrib.auth.tokens import default_token_generator
from django.core.cache import cache
from django.core.mail import send_mail
from django.db import IntegrityError, transaction
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.utils import timezone
from django.views.decorators.http import require_POST

from .models import User, PendingRegistration
from .forms import StartRegistrationForm, UserLogin, \
    CompleteRegistrationForm, ResetPasswordForm


def _throttle_key(scope, *parts):
    normalized_parts = [str(part).strip().lower() for part in parts if str(part).strip()]
    return ":".join([scope, *normalized_parts])


def _throttle_is_limited(key, limit):
    return int(cache.get(key, 0)) >= limit


def _throttle_hit(key, window_seconds):
    if not cache.add(key, 1, timeout=window_seconds):
        try:
            cache.incr(key)
        except ValueError:
            cache.set(key, 1, timeout=window_seconds)


def _throttle_reset(*keys):
    for key in keys:
        if key:
            cache.delete(key)

def get_client_ip(request):
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "")


def register(request):
    if request.user.is_authenticated:
        return redirect("home")
    
    if request.method == "POST":
        if request.session.session_key is None:
            request.session.create()
        form = StartRegistrationForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            ip_key = _throttle_key("register.ip", get_client_ip(request))
            session_key = _throttle_key("register.session", request.session.session_key)
            if (
                _throttle_is_limited(ip_key, settings.REGISTRATION_RATE_LIMIT_ATTEMPTS)
                or _throttle_is_limited(session_key, settings.REGISTRATION_RATE_LIMIT_ATTEMPTS)
            ):
                form.add_error(None, "Превышен лимит по количеству попыток")
            else:
                _throttle_hit(ip_key, settings.REGISTRATION_RATE_LIMIT_WINDOW_SECONDS)
                _throttle_hit(session_key, settings.REGISTRATION_RATE_LIMIT_WINDOW_SECONDS)
                if not User.objects.filter(email__iexact=email).exists():
                    pending, _ = PendingRegistration.objects.update_or_create(
                        email=email,
                        defaults={
                            "token": uuid.uuid4(),
                            "expires_at": timezone.now() + timedelta(hours=24),
                        },
                    )
                    send_verification_email(request, pending)
                return redirect("verify_email_sent")
    else:
        form = StartRegistrationForm()

    return render(request, "accounts/register.html", {"form": form})


def reset_password(request):
    email_sent = False
    if request.method == "POST":
        form = ResetPasswordForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            ip_key = _throttle_key("reset.ip", get_client_ip(request))
            email_key = _throttle_key("reset.email", email)
            if (
                _throttle_is_limited(ip_key, settings.PASSWORD_RESET_RATE_LIMIT_ATTEMPTS)
                or _throttle_is_limited(email_key, settings.PASSWORD_RESET_RATE_LIMIT_ATTEMPTS)
            ):
                form.add_error(None, "Слишком много запросов на сброс. Попробуйте позже.")
            else:
                _throttle_hit(ip_key, settings.PASSWORD_RESET_RATE_LIMIT_WINDOW_SECONDS)
                _throttle_hit(email_key, settings.PASSWORD_RESET_RATE_LIMIT_WINDOW_SECONDS)
                user = User.objects.filter(email__iexact=email, is_google_auth=False).first()
                if user is not None:
                    uidb64 = urlsafe_base64_encode(force_bytes(user.pk))
                    token = default_token_generator.make_token(user)
                    reset_link = request.build_absolute_uri(
                        reverse("complite_reset_password", kwargs={"uidb64": uidb64, "token": token})
                    )
                    send_mail(
                        "Сброс пароля",
                        f"Ссылка на сброс пароля: {reset_link}",
                        from_email=None,
                        recipient_list=[email],
                        fail_silently=False,
                    )
                email_sent = True
                form = ResetPasswordForm()
    else:
        form = ResetPasswordForm()
    return render(request, "password_reset.html", {"form": form, "email_sent": email_sent})


def complite_reset_password(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and default_token_generator.check_token(user, token):
        if request.method == "POST":
            form = SetPasswordForm(user, request.POST)
            if form.is_valid():
                form.save()
                return render(request, "password_reset_complete.html")
        else:
            form = SetPasswordForm(user)
        return render(request, "password_reset_confirm.html", {"form": form})
    else:
        return render(request, "password_reset_invalid.html")


def login_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        normalized_email = (request.POST.get("username") or "").strip().lower()
        ip_key = _throttle_key("login.ip", get_client_ip(request))
        account_key = _throttle_key("login.account", normalized_email)
        form = UserLogin(request, data=request.POST)
        if (
            _throttle_is_limited(ip_key, settings.LOGIN_RATE_LIMIT_ATTEMPTS)
            or _throttle_is_limited(account_key, settings.LOGIN_RATE_LIMIT_ATTEMPTS)
        ):
            form.add_error(None, "Слишком много попыток входа. Попробуйте позже.")
        elif form.is_valid():
            _throttle_reset(ip_key, account_key)
            login(request, form.get_user())
            return redirect("home")
        else:
            _throttle_hit(ip_key, settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS)
            _throttle_hit(account_key, settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS)
    else:
        form = UserLogin(request)

    return render(request, "accounts/login.html", {"form": form})

@require_POST
def logout_view(request):
    logout(request)
    return redirect("home")

def send_verification_email(request, pending):
    verify_url = request.build_absolute_uri(
        reverse("verify_email", kwargs={"token": str(pending.token)})
    )
    message = render_to_string(
        "accounts/email/verify_email.txt",
        {
            "email": pending.email,
            "verify_url": verify_url
        }
    )
    send_mail(
        subject="Подтверждение email",
        message=message,
        from_email=None,
        recipient_list=[pending.email],
        fail_silently=False,
    )

def verify_email_sent(request):
    return render(request, "accounts/verify_email_sent.html")

def verify_email(request, token):
    pending = get_object_or_404(PendingRegistration, token=token)
    if pending.expired:
        pending.delete()
        return render(request, "accounts/verify_email_invalid.html")
    
    request.session["verified_registration_token"] = str(pending.token)
    return redirect("complete_registration")

def complete_registration(request):
    token = request.session.get("verified_registration_token")
    if not token:
        return redirect("register")
    
    pending = get_object_or_404(PendingRegistration, token=token)

    if pending.expired:
        pending.delete()
        request.session.pop("verified_registration_token", None)
        return redirect("register")
    if request.method == "POST":
        form = CompleteRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.email = pending.email
            user.is_active = True
            try:
                with transaction.atomic():
                    user.save()
                    pending.delete()
            except IntegrityError:
                form.add_error(None, "Аккаунт с таким email уже был создан.")
            else:
                request.session.pop("verified_registration_token", None)
                login(request, user)
                return redirect("home")
    else:
        form = CompleteRegistrationForm()
    return render(
        request,
        "accounts/complete_registration.html",
        {"form": form, "email": pending.email},
    )
