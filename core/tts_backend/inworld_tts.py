import base64
from pathlib import Path
import requests

from core.tts_backend.tts_config import DEFAULT_SPEAKERS
from core.utils.config_utils import load_key

url = "https://api.inworld.ai/tts/v1/voice"

def inworld_tts(text: str, save_path: str | Path, speaker_id: str = None):
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    if speaker_id:
        character_dict = load_key("inworld_tts.speakers")
        voice_id = character_dict[speaker_id]
    else:
        voice_id = DEFAULT_SPEAKERS["language_code"]
        
    
    payload = {
        "text": text,
        "voice_id": voice_id,
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
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        audio_content = base64.b64decode(result["audioContent"])
    except Exception as e: 
        print(e)
    with open(save_path, "wb") as f:
        f.write(audio_content)