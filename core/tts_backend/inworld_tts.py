import base64
from pathlib import Path
import requests
from core.tts_backend.get_tts_coef import get_coefficient_tts
from core.tts_backend.tts_config import DEFAULT_SPEAKERS
from core.utils.config_utils import load_key

url = "https://api.inworld.ai/tts/v1/voice"

def inworld_tts(text: str, save_path: str | Path, duration: float, speaker_id: str = None):
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    language_code = load_key("language_code")
    if speaker_id:
        character_dict = load_key("inworld_tts.speakers")
        voice_id = character_dict[speaker_id]
    else:
        voice_id = DEFAULT_SPEAKERS[language_code]
    speaking_rate = get_coefficient_tts(text, language_code, duration)
    payload = {
        "text": text,
        "voice_id": voice_id,
        "model_id": "inworld-tts-1.5-max",
        "audio_config": {
            "audio_encoding": "LINEAR16",
            "sample_rate_hertz": 48000,
        },
        "speakingRate": speaking_rate,
    }
    headers = {
        "Authorization": f"Basic {load_key('inworld_tts.api_key')}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    result = response.json()
    audio_content = base64.b64decode(result["audioContent"])
    with open(save_path, "wb") as f:
        f.write(audio_content)