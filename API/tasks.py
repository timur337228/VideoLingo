from API.query_models import PipelineInput
from API.celery_app import celery
from core.utils.config_utils import update_key
from main import main
from API.s3 import s3
import shutil
import os

LANGUAGE_NAMES = {
    "en": "English",
    "ru": "Русский",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "es": "Español",
    "ja": "日本語",
    "zh": "中文",
}

@celery.task
def run_pipeline_task(data: dict):
    data["target_language"] = LANGUAGE_NAMES[data["language_code"]]
    save_as = data.pop("save_as")
    for key, value in data.items():
        update_key(key, value)
        
    main()
    video_path = s3.upload_file("output/output_dub.mp4", save_as)

    shutil.rmtree("output")
    os.mkdir("output")

    return video_path