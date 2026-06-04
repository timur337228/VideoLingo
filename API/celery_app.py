from celery import Celery

PIPELINE_QUEUE = "pipeline"

celery = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["API.tasks"],
)

celery.conf.task_track_stared = True
celery.conf.task_default_queue = PIPELINE_QUEUE
celery.conf.task_routes = {
    "API.tasks.run_pipeline_task": {"queue": PIPELINE_QUEUE},
}
celery.conf.worker_prefetch_multiplier = 1
celery.conf.worker_concurrency = 1
