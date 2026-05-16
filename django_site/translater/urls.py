from django.urls import path
from translater.views import upload_video, translate_status

urlpatterns = [
    path("upload-video/", upload_video, name="upload_video"),
    path("translate-status/<int:video_id>/", translate_status, name="translate_status"),
]
