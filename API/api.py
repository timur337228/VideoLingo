from celery.result import AsyncResult
from fastapi import FastAPI
from API.tasks import run_pipeline_task
from API.query_models import PipelineInput
from API.celery_app import PIPELINE_QUEUE, celery
from API.s3 import s3


app = FastAPI()

@app.post("/run-pipeline")
def run_pipeline(data: PipelineInput):
    task = run_pipeline_task.apply_async(
        args=[data.model_dump()],
        queue=PIPELINE_QUEUE,
    )
    return {"task_id": task.id}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    task = AsyncResult(task_id, app=celery)
    raw_result = task.result
    output = {
        "status": task.status,
        "result": raw_result,
        "video_url": None,
        "src_url": None,
        "trans_url": None,
        "src_trans_url": None,
        "trans_src_url": None,
        "dub_audio_url": None,
        "background_audio_url": None,
    }

    if task.status == "SUCCESS" and isinstance(raw_result, str):
        try:
            result = s3.resolve_key(raw_result)
            files = set(s3.list_files(result))
            output["result"] = result
            artifact_names = {
                "video_url": "output_dub.mp4",
                "src_url": "src.srt",
                "trans_url": "trans.srt",
                "src_trans_url": "src_trans.srt",
                "trans_src_url": "trans_src.srt",
                "dub_audio_url": "dub.mp3",
                "background_audio_url": "background.mp3",
            }
            for field_name, file_name in artifact_names.items():
                object_key = f"{result}/{file_name}"
                if object_key in files:
                    output[field_name] = s3.signed_url(object_key)

        except Exception:
            output["result"] = raw_result

    return output
