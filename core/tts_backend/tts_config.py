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

FEMALE_GENDER_LABEL = "female"
VOICE_INDEX = {}
SPEAKERS_VOICE = {
    "en": {
        "male": ["Blake",
                 "Clive",
                 "Hades",
                 "Jason",
                 "Mark",
                 "Reed",
                 "Theodore",],
        "female": ["Ashley",
                   "Eleanor",
                   "Hana",
                   "Luna",
                   "Olivia",
                   "Sarah",
                   "Sophie",]
    },
    "ru": {
        "male": [
            "Dmitry",
            "Nikolai",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-b1c1c082",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-b0bfa86f",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-5ad46e0b",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-487e2c97",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-0cb12e78",
        ],
        "female": [
            "Svetlana",
            "Elena",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-47eb37e0",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-7f972cae",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-52dfb9f8",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-63cb52c1",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-a96157df",
        ],
    },
    "de": {
        "male": [
            "Josef",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-c83f5cd8",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-a6243507",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-72472f52",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-e32ea3a2",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-5e5b5b82",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-319aebe4",
        ],
        "female": [
            "Johanna",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-3398bc50",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-fc25db0d",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-c24ad8d0",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-83d33a50",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-e4cf9337",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-abcea90b",
        ],
    },
    "zh": {
        "male": [
            "Ming",
            "Yichen",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-8b88f597",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-efcba425",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-4a66a030",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-95af4625",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-af69a1e8",
        ],
        "female": [
            "Jing",
            "Mei",
            "Xiaoyin",
            "Xiaoyin",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-6a721600",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-5b3405af",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-e3de3218",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-41d6fec4",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-59f7d318",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-1c936644",
        ],
    },
    "fr": {
        "male": [
            "Alain",
            "Étienne",
            "Mathieu",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-c159a984",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-241c658e",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-0bfac39e",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-f9b38f95",
        ],
        "female": [
            "Hélène",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-54e5bda9",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-b0faa3c2",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-a060a155",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-7fdde702",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-b85ab983",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-af999658",
        ],
    },
    "es": {
        "male": [
            "Mateo",
            "Rafael",
            "Diego",
            "Mauricio",
            "Miguel",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-4d57810e",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-b0aaf2c8",
        ],
        "female": [
            "Sofia",
            "Camila",
            "Lupita",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-29c86b02",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-54c63c58",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-bdf596a9",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-5f0b1af1",
        ],
    },
    "ja": {
        "male": [
            "Haruto",
            "Satoshi",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-bbdfd6f7",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-df6ab453",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-a0dc047f",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-8db37ca7",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-39832191",
        ],
        "female": [
            "Asuka",
            "Hina",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-22acee03",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-acd3a7d6",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-4ef9a3bb",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-a6aceb36",
            "default-e2wua3xkbjqhz5qigoakxw__design-voice-9a3c37d8",
        ],
    },
}

def get_voices(label):
    language = load_key("language_code")
    key = (language, label)
    voices = SPEAKERS_VOICE[language][label]

    i = VOICE_INDEX.get(key, 0)
    if i < len(voices):
        voice = voices[i]
    else:
        tail = voices[:-1][::-1] or voices
        voice = tail[(i - len(voices)) % len(tail)]
    VOICE_INDEX[key] = i + 1
    return voice


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

def speakers_send(is_gender_classification: bool = False):
    if not is_gender_classification:
        create_example_tts_file()
    result = {}
    for file in os.listdir(_MERGED_AUDIO_DIR):
        file = Path(file)
        speaker_id = file.name.replace("_merged.wav", "")
        if speaker_id not in result:
            full_path = os.path.join(_MERGED_AUDIO_DIR, file)
            gender = classify_audio(full_path)
            if is_gender_classification and gender != FEMALE_GENDER_LABEL:
                continue
            tag = get_voices(gender) if not is_gender_classification else gender
            result[speaker_id] = tag
    tts_method = load_key("tts_method")
    label = "genders_speakers" if is_gender_classification else f"{tts_method}.speakers"
    update_key(label, result)

def create_example_tts_file():
    os.makedirs(_MERGED_AUDIO_DIR, exist_ok=True)
    get_file_name = lambda: f"{_AUDIO_REFERS_DIR}/{number}.wav"
    speakers = {}
    df = pd.read_csv(_8_1_AUDIO_TASK)
    for i in range(len(df)):
        speaker = df["speaker_id"][i]
        number = df["number"][i]
        if speaker in speakers:
            if len(speakers[speaker]) < 10:
                speakers[speaker].append(get_file_name())
        else:
            speakers[speaker] = [get_file_name()]
    for speaker in speakers:
        files = speakers[speaker]
        result = AudioSegment.empty()
        for f in files:
            result += AudioSegment.from_wav(f)
        result.export(f"{_MERGED_AUDIO_DIR}/{speaker}_merged.wav", format="wav")
