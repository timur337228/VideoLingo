import platform
import subprocess
import os

import cv2
import numpy as np
from rich.console import Console

from core._1_ytdlp import find_video_files
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'

TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'


BACKGROUND_MUSIC_MODE = "background_music"
ORIGINAL_AUDIO_MODE = "original_audio"
DUB_BACKGROUND_AUDIO_KEY = "dub_background_audio"


def get_dub_background_audio_mode():
    try:
        mode = load_key(DUB_BACKGROUND_AUDIO_KEY)
    except KeyError:
        return BACKGROUND_MUSIC_MODE

    normalized = str(mode or BACKGROUND_MUSIC_MODE).strip().lower()
    aliases = {
        "background": BACKGROUND_MUSIC_MODE,
        "music": BACKGROUND_MUSIC_MODE,
        "demucs": BACKGROUND_MUSIC_MODE,
        "background_music": BACKGROUND_MUSIC_MODE,
        "original": ORIGINAL_AUDIO_MODE,
        "original_audio": ORIGINAL_AUDIO_MODE,
        "source": ORIGINAL_AUDIO_MODE,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported {DUB_BACKGROUND_AUDIO_KEY}: {mode!r}. "
            f"Use '{BACKGROUND_MUSIC_MODE}' or '{ORIGINAL_AUDIO_MODE}'."
        )
    return aliases[normalized]


def merge_video_audio():
    """Merge video and audio, and reduce video volume"""
    VIDEO_FILE = find_video_files()
    background_mode = get_dub_background_audio_mode()
    
    # Merge video and audio with translated subtitles
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    
    if load_key("burn_subtitles_dub"):
        subtitle_filter = (
            ","
            f"subtitles={DUB_SUB_FILE}:force_style='FontSize={TRANS_FONT_SIZE},"
            f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
            f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
            f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
        )
    else:
        subtitle_filter = ""

    video_filter = (
        f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
        f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2{subtitle_filter}[v];'
    )

    if background_mode == ORIGINAL_AUDIO_MODE:
        rprint("[cyan]🎚️ Using original video audio as dubbing background.[/cyan]")
        cmd = ['ffmpeg', '-y', '-i', VIDEO_FILE, '-i', DUB_AUDIO]
        audio_filter = '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=3[a]'
    else:
        background_file = _BACKGROUND_AUDIO_FILE
        if not os.path.exists(background_file):
            raise FileNotFoundError(
                f"{background_file} was not found. "
                f"Use {DUB_BACKGROUND_AUDIO_KEY}: '{ORIGINAL_AUDIO_MODE}' "
                "or enable Demucs to create background music."
            )
        rprint("[cyan]🎚️ Using Demucs background music as dubbing background.[/cyan]")
        cmd = ['ffmpeg', '-y', '-i', VIDEO_FILE, '-i', background_file, '-i', DUB_AUDIO]
        audio_filter = '[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3[a]'
    
    cmd.extend(['-filter_complex', f'{video_filter}{audio_filter}'])

    if load_key("ffmpeg_gpu"):
        rprint("[bold green]Using GPU acceleration...[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]', '-c:v', 'h264_nvenc'])
    else:
        cmd.extend(['-map', '[v]', '-map', '[a]'])
    
    cmd.extend(['-c:a', 'aac', '-b:a', '96k', DUB_VIDEO])
    
    subprocess.run(cmd)
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

if __name__ == '__main__':
    merge_video_audio()
