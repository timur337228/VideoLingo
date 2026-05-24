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
    output = {
        "status": task.status,
        "result": raw_result,
        "video_url": None,
        "src_url": None,
        "trans_url": None,
        "src_trans_url": None,
        "trans_src_url": None,
    }

    if task.status == "SUCCESS" and isinstance(raw_result, str):
        try:
            result = s3.resolve_key(raw_result)
            output["result"] = result
            output["video_url"] = s3.signed_url(f"{result}/output_dub.mp4")
            output["src_url"] = s3.signed_url(f"{result}/src.srt")
            output["trans_url"] = s3.signed_url(f"{result}/trans.srt")
            output["src_trans_url"] = s3.signed_url(f"{result}/src_trans.srt")
            output["trans_src_url"] = s3.signed_url(f"{result}/trans_src.srt")

        except Exception:
            result = raw_result

    return output