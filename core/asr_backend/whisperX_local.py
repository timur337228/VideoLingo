import os
import warnings
import time
import subprocess
import torch
import functools

warnings.filterwarnings("ignore")

# =============================================================================
# Compatibility shim — applied BEFORE importing whisperx
# =============================================================================

# torch.load: default weights_only=False for pyannote checkpoints
# PyTorch >=2.6 changed torch.load default to weights_only=True.
# pyannote checkpoints contain omegaconf objects that fail the safety check.
# Monkey-patch torch.load to default to weights_only=False (matching <2.6
# behavior).  This is safe here because all model files come from trusted
# sources (HuggingFace / pyannote).
_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# =============================================================================
# Now safe to import whisperx and the rest of the application
# =============================================================================
import whisperx
from whisperx.audio import load_audio as _whisperx_load_audio, SAMPLE_RATE as _WHISPERX_SR
from rich import print as rprint
from core.utils import *
from core.utils.models import *
MODEL_DIR = load_key("model_dir")

@except_handler("failed to check hf mirror", default_return=None)
def check_hf_mirror():
    mirrors = {'Official': 'huggingface.co', 'Mirror': 'hf-mirror.com'}
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float('inf')
    rprint("[cyan]🔍 Checking HuggingFace mirrors...[/cyan]")
    for name, domain in mirrors.items():
        if os.name == 'nt':
            cmd = ['ping', '-n', '1', '-w', '3000', domain]
        else:
            cmd = ['ping', '-c', '1', '-W', '3', domain]
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        response_time = time.time() - start
        if result.returncode == 0:
            if response_time < best_time:
                best_time = response_time
                fastest_url = f"https://{domain}"
            rprint(f"[green]✓ {name}:[/green] {response_time:.2f}s")
    if best_time == float('inf'):
        rprint("[yellow]⚠️ All mirrors failed, using default[/yellow]")
    rprint(f"[cyan]🚀 Selected mirror:[/cyan] {fastest_url} ({best_time:.2f}s)")
    return fastest_url

def init_whisperx():
    mirror = check_hf_mirror()
    os.environ['HF_ENDPOINT'] = mirror if mirror else "https://huggingface.co"
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rprint(f"🚀 Starting WhisperX using device: {device} ...")
    
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = 16 if gpu_mem >= 8 else 2
        rprint(f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB, [cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️")
    else:
        batch_size = 1
    
    if WHISPER_LANGUAGE == 'zh':
        model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
        local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct-fasterwhisper")
    else:
        model_name = load_key("whisper.model")
        local_model = os.path.join(MODEL_DIR, model_name)
        
    if os.path.exists(local_model):
        rprint(f"[green]📥 Loading local WHISPER model:[/green] {local_model} ...")
        model_name = local_model
    else:
        rprint(f"[green]📥 Using WHISPER model from HuggingFace:[/green] {model_name} ...")

        vad_options = {
        "vad_onset": load_key("whisper.vad_onset"),
        "vad_offset": load_key("whisper.vad_offset"),
    }
    asr_options = {"temperatures": [0],"initial_prompt": "",}
    whisper_language = None if 'auto' in WHISPER_LANGUAGE else WHISPER_LANGUAGE
    rprint("[bold yellow] You can ignore warning of `Model was trained with torch 1.10.0+cu102, yours is 2.0.0+cu118...`[/bold yellow]")
    model = whisperx.load_model(model_name, device, compute_type=load_key("compute_type"), language=whisper_language, vad_options=vad_options, asr_options=asr_options, download_root=MODEL_DIR)
    return model, batch_size, device

AUDIO_CACHE = {}
def get_full_audio(audio_path):
    if audio_path not in AUDIO_CACHE:
        AUDIO_CACHE[audio_path] = _whisperx_load_audio(audio_path, sr=_WHISPERX_SR)
    return AUDIO_CACHE[audio_path]



MODEL, BATCH_SIZE, DEVICE = init_whisperx()
ALL_MODELS = {}

if load_key("whisper.enable_diarization"):
        DIARIZATION_MODEL = whisperx.diarize.DiarizationPipeline(
            token=load_key("hf_token"), 
            device=DEVICE
        )


@except_handler("WhisperX processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    WHISPER_LANGUAGE = load_key("whisper.language")
    full_raw = get_full_audio(raw_audio_file)
    full_vocal = get_full_audio(vocal_audio_file)
    
    def load_audio_segment(audio_file, start, end):
        # Use whisperx's ffmpeg-based loader instead of librosa.load() which
        # deadlocks inside Streamlit's ScriptRunner thread.
        
        start_sample = int(start * _WHISPERX_SR)
        end_sample = int(end * _WHISPERX_SR)
        return audio_file[start_sample:end_sample]

    raw_audio_segment = load_audio_segment(full_raw, start, end)
    vocal_audio_segment = load_audio_segment(full_vocal, start, end)
    
    # -------------------------
    # 1. transcribe raw audio
    # -------------------------
    transcribe_start_time = time.time()
    rprint("[bold green]Note: You will see Progress if working correctly ↓[/bold green]")
    result = MODEL.transcribe(raw_audio_segment, batch_size=BATCH_SIZE, print_progress=True)
    transcribe_time = time.time() - transcribe_start_time
    rprint(f"[cyan]⏱️ time transcribe:[/cyan] {transcribe_time:.2f}s")

    # torch.cuda.empty_cache()

    # Save language
    update_key("whisper.language", result['language'])
    if result['language'] == 'zh' and WHISPER_LANGUAGE != 'zh':
        raise ValueError("Please specify the transcription language as zh and try again!")

    # -------------------------
    # 2. align by vocal audio
    # -------------------------
    align_start_time = time.time()
    # Align timestamps using vocal audio
    if result["language"] in ALL_MODELS:
        model_a, metadata = ALL_MODELS[result["language"]]
    else:
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
        ALL_MODELS[result["language"]] = model_a, metadata
    result = whisperx.align(result["segments"], model_a, metadata, vocal_audio_segment, DEVICE, return_char_alignments=False)
    align_time = time.time() - align_start_time
    rprint(f"[cyan]⏱️ time align:[/cyan] {align_time:.2f}s")

    # -------------------------
    # 3. diarization (optional)
    # -------------------------
    if load_key("whisper.enable_diarization"):
        diarize_start_time = time.time()
        diarize_segments = DIARIZATION_MODEL(vocal_audio_segment)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        diarize_time = time.time() - diarize_start_time
        rprint(f"[cyan]⏱️ time diarize:[/cyan] {diarize_time:.2f}s")

    # Free GPU resources again
    # torch.cuda.empty_cache()

    # Adjust timestamps
    for segment in result['segments']:
        segment['start'] += start
        segment['end'] += start
        for word in segment['words']:
            if 'start' in word:
                word['start'] += start
            if 'end' in word:
                word['end'] += start
    return result
