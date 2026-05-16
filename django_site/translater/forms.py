from django import forms

class UploadVideo(forms.Form):
    file = forms.FileField(label="Видео")
    CHOICES_LANGUAGE = [
    ('en', 'Английский 🇺🇸'),
    ('ru', 'Русский 🇷🇺'),
    ('fr', 'Французский 🇫🇷'),
    ('de', 'Немецкий 🇩🇪'),
    ('it', 'Итальянский 🇮🇹'),
    ('es', 'Испанский 🇪🇸'),
    ('ja', 'Японский 🇯🇵'),
    ('zh', 'Китайский 🇨🇳'),
]
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
        self.fields["file"].widget.attrs.update({"class": "form-input", "accept": "video/*"})
        self.fields["language"].widget.attrs.update({"class": "form-input"})
        self.fields["volume"].widget.attrs.update({"class": "form-input"})
