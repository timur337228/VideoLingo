from celery.result import AsyncResult
from fastapi import FastAPI
from API.tasks import run_pipeline_task
from API.query_models import PipelineInput
from API.celery_app import celery
from API.s3 import s3


app = FastAPI()

@app.post("/run-pipeline")
def run_pipeline(data: PipelineInput):
    task = run_pipeline_task.delay(data.model_dump())
    return {"task_id": task.id}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    task = AsyncResult(task_id, app=celery)
    raw_result = task.result
    result = raw_result
    video_url = None

    if task.status == "SUCCESS" and isinstance(raw_result, str):
        try:
            result = s3.resolve_key(raw_result)
            video_url = s3.signed_url(result)
        except Exception:
            result = raw_result

    return {
        "status": task.status,
        "result": result,
        "video_url": video_url,
    }
