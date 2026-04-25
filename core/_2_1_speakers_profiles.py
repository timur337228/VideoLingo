import datetime
import os
import numpy as np
from rich.panel import Panel
from core._8_1_audio_task import time_diff_seconds
from core.utils import *
from core.utils.models import *
import pandas as pd
import soundfile as sf
from core.asr_backend.demucs_vl import demucs_audio
from core.utils.models import *
from core.tts_backend.tts_config import speakers_send    


def normalize_speaker_id(value):
    if pd.isna(value):
        return None
    return str(value)


def extract_audio(audio_data, sr, start_time, end_time, out_file = None, is_save: bool = True):
    """Simplified audio extraction function"""
    start = int(start_time * sr)
    end = int(end_time * sr)
    if is_save and out_file is not None:
        sf.write(out_file, audio_data[start:end], sr)
    return audio_data[start:end]

def group_split(df):
    rows = [row for _, row in df.iterrows()]
    if not rows:
        return []
    groups = [
        [rows[0]],]
    index = 0
    for row in rows[1:]:
        if groups[index][-1]["speaker_id"] != row["speaker_id"] or \
            float(row["start"]) - float(groups[index][-1]["end"]) > 2:
            index += 1
            groups.append([])
        groups[index].append(row)
    return groups

def get_gender_speakers():

    update_key("is_gender_translate", load_key("language_code") in load_key("language_with_gender"))
    if not load_key("whisper.enable_diarization") or not load_key("is_gender_translate"):
        return 
    demucs_audio()

    os.makedirs(_MERGED_AUDIO_DIR, exist_ok=True)
    
    df = pd.read_csv(_2_CLEANED_CHUNKS)
    if "speaker_id" not in df.columns or df.empty:
        update_key("genders_speakers", {})
        rprint(Panel("No diarization speaker data found, skipping gender detection.", title="Info", border_style="yellow"))
        return
    df["speaker_id"] = df["speaker_id"].apply(normalize_speaker_id)
    data, sr = sf.read(_VOCAL_AUDIO_FILE)

    # speakers logic
    speakers = load_key("all_speakers") or []
    speakers = [speaker for speaker in (normalize_speaker_id(s) for s in speakers) if speaker]
    if not speakers:
        speakers = [speaker for speaker in df["speaker_id"].dropna().unique().tolist()]
        update_key("all_speakers", speakers)
    speakers = {speaker: [] for speaker in speakers}
    groups = group_split(df)

    for group in groups:
        for row in group:
            speaker = normalize_speaker_id(row["speaker_id"])
            if not speaker:
                continue
            if speaker not in speakers:
                speakers[speaker] = []
            if len(speakers[speaker]) < 10:
                speakers[speaker].append(row)
            
    for speaker in speakers:
        audio_data = []
        for row in speakers[speaker]:
            audio_data.append(extract_audio(data, sr, row['start'], row['end']))
        if not audio_data:
            continue
        sf.write(f"{_MERGED_AUDIO_DIR}/{speaker}_merged.wav", np.concatenate(audio_data), sr)
            
    speakers_send(True)
    for name in os.listdir(_MERGED_AUDIO_DIR):
        path = os.path.join(_MERGED_AUDIO_DIR, name)
        os.remove(path)
    rprint(Panel(f"Audio saved to {_MERGED_AUDIO_DIR}", title="Success", border_style="green"))

if __name__ == "__main__":
    get_gender_speakers()
