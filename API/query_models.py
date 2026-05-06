from pydantic import BaseModel

class PipelineInput(BaseModel):
    save_as: str
    target_language: str
    dub_background_audio: str = "original_audio"
    dub_background_volume_percent: int | None = 12
    burn_subtitles_dub: bool = False

    
