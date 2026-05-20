import os
import uuid
import shutil
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
import requests
from django.shortcuts import get_object_or_404
from .forms import UploadVideo
from .models import Video


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
            
            payload = {
                "save_as": f"tmp/{user.pk}_{uuid.uuid4()}.mp4",
                "language_code": form.cleaned_data["language"],
                "dub_background_audio": dub_background_audio,
                "dub_background_volume_percent": form.cleaned_data["volume"],
                "burn_subtitles_dub": form.cleaned_data["is_sub"],
            }
            try:
                response = requests.post(
                    f"{settings.API_BASE_URL}/run-pipeline",
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                video = Video.objects.create(
                    user=user,
                    task_id=response.json()["task_id"],
                    status="PENDING",
                )
                return redirect("translate_status", video_id=video.pk)
            except requests.RequestException as e:
                return render(request, "translater/error_as_upload.html", {"error": str(e)})
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

    context = {
        "video": video,
        "api_status": payload.get("status"),
        "api_result": result,
        "video_playback_url": payload.get("video_url"),
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
