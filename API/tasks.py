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
    try:
        data["target_language"] = LANGUAGE_NAMES[data["language_code"]]
        save_dir = data.pop("save_dir")
        for key, value in data.items():
            update_key(key, value)
        main()
        AUDIO_FILE_NAMES = {
            "output_dub.mp4",
            "src.srt",
            "trans.srt",
            "src_trans.srt",
            "trans_src.srt",
        }
        for file in AUDIO_FILE_NAMES:
            s3.upload_file(f"output/{file}", f"{save_dir}/{file}")
    finally:
        shutil.rmtree("output")
        os.mkdir("output")

    return save_dir