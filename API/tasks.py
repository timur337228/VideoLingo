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
        artifact_paths = {
            "output_dub.mp4": "output/output_dub.mp4",
            "src.srt": "output/src.srt",
            "trans.srt": "output/trans.srt",
            "src_trans.srt": "output/src_trans.srt",
            "trans_src.srt": "output/trans_src.srt",
            "dub.mp3": "output/dub.mp3",
            "background.mp3": "output/audio/background.mp3",
        }
        for remote_name, local_path in artifact_paths.items():
            if os.path.exists(local_path):
                s3.upload_file(local_path, f"{save_dir}/{remote_name}")
    finally:
        shutil.rmtree("output")
        os.mkdir("output")

    return save_dir
