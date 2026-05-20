from django.urls import path
from translater.views import get_my_videos, translate_status, upload_video

urlpatterns = [
    path("upload-video/", upload_video, name="upload_video"),
    path("my-videos/", get_my_videos, name="my_videos"),
    path("video/<int:video_id>/", translate_status, name="translate_status"),
]
