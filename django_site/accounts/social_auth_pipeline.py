from django.contrib.auth import get_user_model
from social_core.exceptions import AuthFailed

def google_require_email(backend, details, response, user=None, **kwargs):
    email = (details.get("email") or response.get("email") or "").strip().lower()
    if not email:
        raise AuthFailed("Google не вернул email")
    details["email"] = email
    return {"details": details}


def google_associate_by_email(backend, details, user=None, **kwargs):
    if user:
        return {"user": user}
    email = details.get("email")
    if not email:
        return
    User = get_user_model()
    existing_user = User.objects.filter(email__iexact=email).first()
    if existing_user:
        return {"user": existing_user}
    
def google_create_user(backend, details, user=None, **kwargs):
    if user:
        return {"user": user}

    email = details["email"]
    User = get_user_model()
    user = User.objects.create_user(email=email, password=None, is_google_auth=True)
    return {"user": user}
