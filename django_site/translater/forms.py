from django import forms
from django.conf import settings

from django_site.languages import LANGUAGE_CHOICES

class UploadVideo(forms.Form):
    file = forms.FileField(label="Видео")
    CHOICES_LANGUAGE = LANGUAGE_CHOICES
    language = forms.ChoiceField(choices=CHOICES_LANGUAGE, label="Выберите язык")
    is_sub = forms.BooleanField(label='Вшить субтитры в итоговое видео', required=False)
    volume = forms.IntegerField(
        label="Громкость фонового звука",
        min_value=0,
        max_value=100,
        initial=12,
        widget=forms.NumberInput(attrs={'type': 'range', 'step': '1'})
    )
    is_del_vocal = forms.BooleanField(label='Убрать оригинальный голос с фона', required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        accept_types = ",".join(settings.ALLOWED_VIDEO_EXTENSIONS)
        self.fields["file"].widget.attrs.update({"class": "form-input", "accept": accept_types})
        self.fields["language"].widget.attrs.update({"class": "form-input"})
        self.fields["volume"].widget.attrs.update({"class": "form-input"})

    def clean_file(self):
        file = self.cleaned_data["file"]
        filename = (file.name or "").lower()
        if not filename.endswith(settings.ALLOWED_VIDEO_EXTENSIONS):
            raise forms.ValidationError("Допустимы только видеофайлы.")

        content_type = (getattr(file, "content_type", "") or "").lower()
        if content_type and content_type not in settings.ALLOWED_VIDEO_CONTENT_TYPES:
            raise forms.ValidationError("Неподдерживаемый тип файла.")

        if file.size > settings.MAX_VIDEO_UPLOAD_SIZE_BYTES:
            raise forms.ValidationError(
                f"Размер файла превышает лимит {settings.MAX_VIDEO_UPLOAD_SIZE_MB} МБ."
            )

        return file
