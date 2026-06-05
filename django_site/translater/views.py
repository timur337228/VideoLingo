import os
import shutil
import uuid
from datetime import timedelta
import logging

import requests
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect, render

from billing.utils import (
    InsufficientBalanceError,
    confirm_video_charge,
    refund_video_charge,
    reserve_video_minutes,
    seconds_to_charge,
)
from .forms import UploadVideo
from .models import Video
from .utils import get_video_duration_seconds


logger = logging.getLogger(__name__)


def _get_video_upload_dir(video_id):
    return os.path.join(settings.UPLOADS_DIR, str(video_id))


def _save_uploaded_source_file(file, video_id):
    upload_dir = _get_video_upload_dir(video_id)
    os.makedirs(upload_dir, exist_ok=True)

    _, ext = os.path.splitext(file.name)
    source_filename = f"source{ext.lower()}"
    source_path = os.path.join(upload_dir, source_filename)

    with open(source_path, "wb+") as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    return source_path


def _delete_video_upload_dir(video_id):
    shutil.rmtree(_get_video_upload_dir(video_id), ignore_errors=True)


@login_required
def upload_video(request):
    insufficient_balance = None

    if request.method == "POST":
        form = UploadVideo(request.POST, request.FILES)
        if form.is_valid():
            dub_background_audio = {
                True: "background_music", 
                False: "original_audio"
                }[form.cleaned_data["is_del_vocal"]]
            file = request.FILES["file"]
            user = request.user
            video = None
            source_path = None
            duration_seconds = 0
            try:
                video = Video.objects.create(
                    user=user,
                    status="QUEUED",
                    task_id=f"pending-{uuid.uuid4()}",
                )
                source_path = _save_uploaded_source_file(file, video.pk)
                duration_seconds = get_video_duration_seconds(source_path)
                video.duration = timedelta(seconds=duration_seconds)
                video.duration_seconds = duration_seconds
                video.save(update_fields=["duration", "duration_seconds"])
                reserve_video_minutes(user, duration_seconds, video)
                payload = {
                    "save_dir": f"users/{user.pk}/videos/{video.pk}",
                    "source_path": source_path,
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
                    _delete_video_upload_dir(video.pk)
                    video.delete()
                required_minutes = seconds_to_charge(duration_seconds) // 60 if duration_seconds else 0
                insufficient_balance = {
                    "message": str(e),
                    "required_minutes": required_minutes,
                    "available_minutes": user.available_minutes,
                }
            except requests.RequestException as e:
                logger.warning("Failed to enqueue translation task for user_id=%s: %s", user.pk, e)
                if video is not None:
                    refund_video_charge(video, "Возврат: задача не была поставлена в очередь")
                    _delete_video_upload_dir(video.pk)
                    video.delete()
                return render(
                    request,
                    "translater/error_as_upload.html",
                    {"error": "Не удалось связаться с сервисом обработки. Попробуйте позже."},
                )
            except Exception as e:
                logger.exception("Failed to prepare uploaded video for user_id=%s", user.pk)
                if video is not None:
                    refund_video_charge(video, "Возврат после ошибки запуска обработки")
                    _delete_video_upload_dir(video.pk)
                    video.delete()
                form.add_error(None, "Не удалось подготовить видео к отправке. Попробуйте другой файл.")
    else:
        form = UploadVideo()
    return render(
        request,
        "translater/upload_video.html",
        {
            "form": form,
            "insufficient_balance": insufficient_balance,
        },
    )




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
        logger.warning("Failed to fetch translation status for video_id=%s: %s", video.pk, e)
        return render(
            request,
            "translater/error_for_translate_status.html",
            {"error": "Не удалось получить статус обработки. Попробуйте обновить страницу позже.", "video": video},
        )

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
