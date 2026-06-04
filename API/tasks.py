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

WORKING_OUTPUT_DIR = "output"


def _reset_working_output_dir():
    shutil.rmtree(WORKING_OUTPUT_DIR, ignore_errors=True)
    os.makedirs(WORKING_OUTPUT_DIR, exist_ok=True)


def _stage_source_video(source_path: str):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source video not found: {source_path}")

    destination = os.path.join(WORKING_OUTPUT_DIR, os.path.basename(source_path))
    shutil.copy2(source_path, destination)
    return destination


def _cleanup_uploaded_source(source_path: str):
    source_dir = os.path.dirname(source_path)
    if source_dir:
        shutil.rmtree(source_dir, ignore_errors=True)


@celery.task
def run_pipeline_task(data: dict):
    source_path = data.pop("source_path")
    try:
        data["target_language"] = LANGUAGE_NAMES[data["language_code"]]
        save_dir = data.pop("save_dir")
        _reset_working_output_dir()
        _stage_source_video(source_path)
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
        _reset_working_output_dir()
        _cleanup_uploaded_source(source_path)

    return save_dir
