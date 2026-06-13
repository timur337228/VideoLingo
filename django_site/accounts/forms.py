from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

User = get_user_model()


class StyledFieldsMixin:
    field_placeholders = {}

    def apply_styles(self):
        for name, field in self.fields.items():
            if isinstance(field.widget, forms.HiddenInput):
                continue
            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs["class"] = "checkbox-input"
            else:
                field.widget.attrs["class"] = "form-input"
            placeholder = self.field_placeholders.get(name)
            if placeholder:
                field.widget.attrs["placeholder"] = placeholder


class StartRegistrationForm(StyledFieldsMixin, forms.Form):
    email = forms.EmailField(label="Email")
    accept_legal = forms.BooleanField(
        label="",
        required=True,
        error_messages={"required": "Нужно принять оферту и политику обработки персональных данных."},
    )

    field_placeholders = {
        "email": "you@example.com",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_styles()
        self.fields["email"].widget.attrs["autocomplete"] = "email"

    def clean_email(self):
        return self.cleaned_data["email"].strip().lower()

class ResetPasswordForm(StyledFieldsMixin, forms.Form):
    email = forms.EmailField(label="Email")

    field_placeholders = {
        "email": "you@example.com",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_styles()
        self.fields["email"].widget.attrs["autocomplete"] = "email"
    
    def clean_email(self):
        return self.cleaned_data["email"].strip().lower()


class CompleteRegistrationForm(StyledFieldsMixin, UserCreationForm):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_styles()
        
    class Meta:
        model = User
        fields = ("password1", "password2")


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

    def clean_username(self):
        return self.cleaned_data["username"].strip().lower()
        
