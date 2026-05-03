import os
import io
import json
import time
import threading
import requests
import librosa
import soundfile as sf
from rich import print as rprint
from core.utils import *
from core.utils.models import *

OUTPUT_LOG_DIR = "output/log"
_AUDIO_CACHE = {}
_AUDIO_CACHE_LOCK = threading.Lock()


def _load_cached_audio(audio_path: str):
    abs_path = os.path.abspath(audio_path)
    stat = os.stat(abs_path)
    signature = (stat.st_mtime_ns, stat.st_size)

    with _AUDIO_CACHE_LOCK:
        cached = _AUDIO_CACHE.get(abs_path)
        if cached and cached["signature"] == signature:
            return cached["audio"]

        audio = librosa.load(abs_path, sr=16000)
        _AUDIO_CACHE[abs_path] = {"signature": signature, "audio": audio}
        return audio


def transcribe_audio_302(raw_audio_path: str, vocal_audio_path: str, start: float = None, end: float = None):
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    LOG_FILE = f"{OUTPUT_LOG_DIR}/whisperx302_{start}_{end}.json"
    if is_cache_enabled() and os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
        
    WHISPER_LANGUAGE = load_key("whisper.language")
    update_key("whisper.language", WHISPER_LANGUAGE)
    url = "https://api.302.ai/302/whisperx"
    
    y, sr = _load_cached_audio(vocal_audio_path)
    audio_duration = len(y) / sr
    
    if start is None or end is None:
        start = 0
        end = audio_duration
        
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    y_slice = y[start_sample:end_sample]
    
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, y_slice, sr, format='WAV', subtype='PCM_16')
    audio_buffer.seek(0)
    
    files = [('audio_input', ('audio_slice.wav', audio_buffer, 'application/octet-stream'))]
    processing_type = "diarize" if load_key("whisper.enable_diarization") else "aligh"
    payload = {"processing_type": processing_type, "language": WHISPER_LANGUAGE, "output": "raw"}
    
    start_time = time.time()
    rprint(f"[cyan]🎤 Transcribing audio with language:  <{WHISPER_LANGUAGE}> ...[/cyan]")
    headers = {'Authorization': f'Bearer {load_key("whisper.whisperX_302_api_key")}'}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    response_json = response.json()
    
    if start is not None:
        for segment in response_json['segments']:
            segment['start'] += start
            segment['end'] += start
            for word in segment.get('words', []):
                if 'start' in word:
                    word['start'] += start
                if 'end' in word:
                    word['end'] += start
    
    if is_cache_enabled():
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(response_json, f, indent=4, ensure_ascii=False)
            print(response_json)
    
    elapsed_time = time.time() - start_time
    rprint(f"[green]✓ Transcription completed in {elapsed_time:.2f} seconds[/green]")
    return response_json

if __name__ == "__main__":  
    result = transcribe_audio_302(_RAW_AUDIO_FILE, _RAW_AUDIO_FILE)
    rprint(result)
