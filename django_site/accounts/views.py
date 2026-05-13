from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.db import IntegrityError, transaction
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.contrib.auth import login, logout
from django.shortcuts import redirect, render
from django.shortcuts import get_object_or_404
from django.core.cache import cache
import uuid
from datetime import timedelta
from django.utils import timezone

from .models import User, PendingRegistration
from .forms import StartRegistrationForm, UserLogin, CompleteRegistrationForm

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
            allowed = cache.add(f"register.ip:{get_client_ip(request)}", True, timeout=120)
            session_key = cache.add(f"register:session:{request.session.session_key}", True, timeout=120)
            if not allowed or not session_key:
                form.add_error(None, "Превышен лимит по количеству попыток")
            else:
                email = form.cleaned_data["email"]
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


def login_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        form = UserLogin(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect("home")
    else:
        form = UserLogin(request)

    return render(request, "accounts/login.html", {"form": form})

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

