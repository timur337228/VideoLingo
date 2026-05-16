from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Video",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("path_to_s3", models.URLField(blank=True, null=True)),
                ("duration", models.TimeField(blank=True, null=True)),
                ("preview", models.URLField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("status", models.CharField(default="PENDING", max_length=32)),
                ("task_id", models.CharField(max_length=255, unique=True)),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="videos", to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
