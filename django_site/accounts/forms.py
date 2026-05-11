from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

User = get_user_model()


class StyledFieldsMixin:
    field_placeholders = {}

    def apply_styles(self):
        for name, field in self.fields.items():
            field.widget.attrs["class"] = "form-input"
            placeholder = self.field_placeholders.get(name)
            if placeholder:
                field.widget.attrs["placeholder"] = placeholder


class UserRegister(StyledFieldsMixin, UserCreationForm):
    email = forms.EmailField(label="Email")

    class Meta:
        model = User
        fields = ("email", "password1", "password2")

    field_placeholders = {
        "email": "you@example.com",
        "password1": "Минимум 8 символов",
        "password2": "Повторите пароль",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_styles()
        self.fields["email"].widget.attrs["autocomplete"] = "email"
        self.fields["password1"].widget.attrs["autocomplete"] = "new-password"
        self.fields["password2"].widget.attrs["autocomplete"] = "new-password"

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("Пользователь с таким email уже существует.")
        return email


class UserLogin(StyledFieldsMixin, AuthenticationForm):
    username = forms.EmailField(label="Email")
    password = forms.CharField(
        label="Пароль",
        strip=False,
        widget=forms.PasswordInput(),
    )

    field_placeholders = {
        "username": "you@example.com",
        "password": "Введите пароль",
    }

    def __init__(self, request=None, *args, **kwargs):
        super().__init__(request=request, *args, **kwargs)
        self.apply_styles()
        self.fields["username"].widget.attrs["autocomplete"] = "email"
        self.fields["password"].widget.attrs["autocomplete"] = "current-password"
