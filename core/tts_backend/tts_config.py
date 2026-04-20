import os
from pathlib import Path
from pydub import AudioSegment
import numpy as np
import pandas as pd
import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from core.utils.config_utils import *
from core.utils.models import *

DEFAULT_SPEAKERS = {
    "ru": "Dmitry",
}

SPEAKERS_VOICE = {
    "ru": {"male": ["Dmitry",
                    "default-hglgrqgxmxsw2sn3ovpfuw__design-voice-5608986b",
                    "Nikolai",
                    "default-hglgrqgxmxsw2sn3ovpfuw__design-voice-920bd5cd",
                    "default-hglgrqgxmxsw2sn3ovpfuw__design-voice-4b7fbee6",],
           "female": ["Svetlana",
                      "default-hglgrqgxmxsw2sn3ovpfuw__design-voice-22abffc4",
                      "Elena",
                      ]}
}

def get_voices(label):
    language = load_key("language_code")
    return SPEAKERS_VOICE[language][label].pop(0)


model_name = "prithivMLmods/Common-Voice-Gender-Detection"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

def classify_audio(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    pred_id = torch.argmax(logits, dim=1).item()
    id2label = { "0": "female", "1": "male" }
    label = model.config.id2label[pred_id]
    return label

def speakers_send():
    create_example_tts_file()
    result = {}
    for file in os.listdir(_MERGED_AUDIO_DIR):
        file = Path(file)
        speaker_id = file.name.replace("_merged.wav", "")
        if speaker_id not in result:
            full_path = os.path.join(_MERGED_AUDIO_DIR, file)
            voice = get_voices(classify_audio(full_path))
            result[speaker_id] = voice
    tts_method = load_key("tts_method")
    update_key(f"{tts_method}.speakers", result)

def create_example_tts_file():
    os.makedirs(_MERGED_AUDIO_DIR, exist_ok=True)
    get_file_name = lambda: f"{_AUDIO_REFERS_DIR}/{number}.wav"
    speakers = {}
    df = pd.read_csv(_8_1_AUDIO_TASK)
    for i in range(len(df)):
        speaker = df["speaker_id"][i]
        number = df["number"][i]
        if speaker in speakers:
            if len(speakers[speaker]) <= 10:
                speakers[speaker].append(get_file_name())
        else:
            speakers[speaker] = [get_file_name()]
    for speaker in speakers:
        files = speakers[speaker]
        result = AudioSegment.empty()
        for f in files:
            result += AudioSegment.from_wav(f)
        result.export(f"{_MERGED_AUDIO_DIR}/{speaker}_merged.wav", format="wav")
