import concurrent.futures
import os

from core.utils import *
from core.asr_backend.demucs_vl import demucs_audio
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, normalize_audio_volume
from core._1_ytdlp import find_video_files
from core.utils.models import *


def _transcribe_segments(ts, raw_audio, vocal_audio, segments, runtime):
    if runtime != "cloud" or len(segments) <= 1:
        return [ts(raw_audio, vocal_audio, start, end) for start, end in segments]

    max_workers = max(1, min(int(load_key("max_workers")), len(segments)))
    rprint(f"[cyan]🎤 Transcribing {len(segments)} cloud ASR segments in parallel with {max_workers} workers...[/cyan]")

    results = [None] * len(segments)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(ts, raw_audio, vocal_audio, start, end): index
            for index, (start, end) in enumerate(segments)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    return results


def _prepare_asr_audio(source_audio):
    return normalize_audio_volume(source_audio, _ASR_NORMALIZED_AUDIO_FILE, format="mp3")


def _cleanup_asr_audio(audio_path):
    if audio_path == _ASR_NORMALIZED_AUDIO_FILE and os.path.exists(audio_path):
        os.remove(audio_path)


@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 1. video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # 2. Demucs vocal separation:
    if load_key("demucs"):
        demucs_audio()
        vocal_audio = _prepare_asr_audio(_VOCAL_AUDIO_FILE)
    else:
        vocal_audio = _prepare_asr_audio(_RAW_AUDIO_FILE)

    # 3. Extract audio
    segments = split_audio(_RAW_AUDIO_FILE)
    
    # 4. Transcribe audio by clips
    runtime = load_key("whisper.runtime")
    if runtime == "local":
        from core.asr_backend.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with local model...[/cyan]")
    elif runtime == "cloud":
        from core.asr_backend.whisperX_302 import transcribe_audio_302 as ts
        rprint("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")
    elif runtime == "elevenlabs":
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts
        rprint("[cyan]🎤 Transcribing audio with ElevenLabs API...[/cyan]")
    else:
        raise ValueError(f"Unsupported whisper.runtime: {runtime}")

    try:
        all_results = _transcribe_segments(ts, _RAW_AUDIO_FILE, vocal_audio, segments, runtime)
    finally:
        _cleanup_asr_audio(vocal_audio)
    
    # 5. Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # 6. Process df
    df = process_transcription(combined_result)
    if load_key("whisper.enable_diarization"):
        speakers = []
        for speaker in df["speaker_id"]:
            if speaker not in speakers:
                speakers.append(speaker)
        update_key("all_speakers", speakers)
    save_results(df)
        
if __name__ == "__main__":
    transcribe()
