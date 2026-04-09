import base64
from pathlib import Path
import requests

from core.utils.config_utils import load_key

url = "https://api.inworld.ai/tts/v1/voice"

def inworld_tts(text: str, save_path: str | Path):
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": text,
        "voice_id": "Dmitry",
        "model_id": "inworld-tts-1.5-max",
        "audio_config": {
            "audio_encoding": "LINEAR16",
            "sample_rate_hertz": 48000,
        },
    }
    headers = {
        "Authorization": f"Basic {load_key('inworld_tts.api_key')}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    audio_content = base64.b64decode(result["audioContent"])
    with open(save_path, "wb") as f:
        f.write(audio_content)