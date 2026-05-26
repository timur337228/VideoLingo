import os
import shutil
import uuid
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect, render
import requests

from billing.utils import (
    InsufficientBalanceError,
    confirm_video_charge,
    refund_video_charge,
    reserve_video_minutes,
)
from .forms import UploadVideo
from .models import Video
from .utils import get_video_duration_seconds


def _reset_output_dir():
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    for item in os.listdir(settings.OUTPUT_DIR):
        path = os.path.join(settings.OUTPUT_DIR, item)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


@login_required
def upload_video(request):
    if request.method == "POST":
        form = UploadVideo(request.POST, request.FILES)
        if form.is_valid():
            dub_background_audio = {
                True: "background_music", 
                False: "original_audio"
                }[form.cleaned_data["is_del_vocal"]]
            file = request.FILES["file"]
            user = request.user
            _reset_output_dir()
            _, ext = os.path.splitext(file.name)
            input_filename = f"source_{user.pk}_{uuid.uuid4().hex}{ext.lower()}"
            input_path = os.path.join(settings.OUTPUT_DIR, input_filename)
            with open(input_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            video = None
            try:
                duration_seconds = get_video_duration_seconds(input_path)
                video = Video.objects.create(
                    user=user,
                    task_id=f"pending-{uuid.uuid4()}",
                    duration=timedelta(seconds=duration_seconds),
                    duration_seconds=duration_seconds,
                )
                reserve_video_minutes(user, duration_seconds, video)
                payload = {
                    "save_dir": f"users/{user.pk}/videos/{video.pk}",
                    "language_code": form.cleaned_data["language"],
                    "dub_background_audio": dub_background_audio,
                    "dub_background_volume_percent": form.cleaned_data["volume"],
                    "burn_subtitles_dub": form.cleaned_data["is_sub"],
                }
                response = requests.post(
                    f"{settings.API_BASE_URL}/run-pipeline",
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()

                video.task_id = response.json()["task_id"]
                video.status = "PENDING"
                video.save(update_fields=["task_id", "status"])
                return redirect("translate_status", video_id=video.pk)
            except InsufficientBalanceError as e:
                if video is not None:
                    video.delete()
                form.add_error(None, str(e))
            except requests.RequestException as e:
                if video is not None:
                    refund_video_charge(video, "Возврат: задача не была поставлена в очередь")
                    video.delete()
                return render(request, "translater/error_as_upload.html", {"error": str(e)})
            except Exception as e:
                if video is not None:
                    refund_video_charge(video, "Возврат после ошибки запуска обработки")
                    video.delete()
                form.add_error(None, f"Не удалось подготовить видео к отправке: {e}")
    else:
        form = UploadVideo()
    return render(request, "translater/upload_video.html", {"form": form})




@login_required
def translate_status(request, video_id):
    try:
        video = get_object_or_404(Video, pk=video_id, user=request.user)
        response = requests.get(
            f"{settings.API_BASE_URL}/status/{video.task_id}",
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as e:
        return render(request, "translater/error_for_translate_status.html", {"error": str(e), "video": video})

    video.status = payload.get("status", video.status)
    result = payload.get("result")
    if video.status == "SUCCESS" and isinstance(result, str):
        video.path_to_s3 = result
    video.save(update_fields=["status", "path_to_s3"])

    if video.status == "SUCCESS":
        confirm_video_charge(video)
    elif video.status in {"FAILURE", "REVOKED"}:
        refund_video_charge(video, "Возврат минут после неуспешной обработки видео")

    artifact_links = [
        {
            "label": "Исходные субтитры",
            "description": "Оригинальная дорожка",
            "url": payload.get("src_url"),
        },
        {
            "label": "Переведенные субтитры",
            "description": "Только перевод",
            "url": payload.get("trans_url"),
        },
        {
            "label": "Source + Translation",
            "description": "Сначала оригинал, потом перевод",
            "url": payload.get("src_trans_url"),
        },
        {
            "label": "Translation + Source",
            "description": "Сначала перевод, потом оригинал",
            "url": payload.get("trans_src_url"),
        },
        {
            "label": "Перевод: аудиодорожка",
            "description": "Голос перевода без видео",
            "url": payload.get("dub_audio_url"),
        },
        {
            "label": "Фоновая дорожка",
            "description": "Фон или музыка без голоса перевода",
            "url": payload.get("background_audio_url"),
        },
    ]
    artifact_links = [item for item in artifact_links if item["url"]]

    context = {
        "video": video,
        "api_status": payload.get("status"),
        "api_result": result,
        "video_playback_url": payload.get("video_url"),
        "artifact_links": artifact_links,
    }
    return render(request, "translater/translate_status.html", context)


@login_required
def get_my_videos(request):
    videos = request.user.videos.order_by("-created_at")
    return render(
        request,
        "translater/my_videos.html",
        {"videos": videos},
    )
