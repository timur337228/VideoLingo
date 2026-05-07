from celery.result import AsyncResult
from fastapi import FastAPI
from API.tasks import run_pipeline_task
from API.query_models import PipelineInput
from API.celery_app import celery


app = FastAPI()

@app.post("/run-pipeline")
def run_pipeline(data: PipelineInput):
    task = run_pipeline_task.delay(data.model_dump())
    return {"task_id": task.id}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    task = AsyncResult(task_id, app=celery)

    return {
        "status": task.status,
        "result": task.result
    }

