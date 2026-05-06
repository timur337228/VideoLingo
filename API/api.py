from fastapi import FastAPI
from core.utils.config_utils import update_key
from API.query_models import PipelineInput
from API.s3 import s3
from main import main
import os
import shutil

app = FastAPI()

@app.post("/run-pipeline")
def run_pipeline(data: PipelineInput):
    for key, value in data.model_dump(exclude={"save_as"}).items():
        update_key(key, value)
    main()
    video_path = s3.upload_file("output/output_dub.mp4", data.save_as)
    shutil.rmtree("output")
    os.mkdir("output")
    return video_path


