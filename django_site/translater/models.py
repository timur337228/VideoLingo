from django.db import models
from accounts.models import User


class Video(models.Model):
    path_to_s3 = models.URLField(null=True, blank=True)
    duration = models.TimeField(null=True, blank=True)
    preview = models.URLField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=32, default="PENDING")
    task_id = models.CharField(max_length=255, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="videos")
