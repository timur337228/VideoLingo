import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from core import *


console = Console()


# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_translate_steps():
    return [
        ("🎙️ WhisperX transcription", _2_asr.transcribe),
        ("✂️ NLP sentence split", _3_1_split_nlp.split_by_spacy),
        ("🧠 Meaning split", _3_2_split_meaning.split_sentences_by_meaning),
        ("📝 Summary and terminology", _4_1_summarize.get_summary),
        ("🌍 Translation", _4_2_translate.translate_all),
        ("📏 Subtitle split", _5_split_sub.split_for_sub_main),
        ("⏱️ Subtitle timestamp alignment", _6_gen_sub.align_timestamp_main),
        ("🎬 Burn subtitles into video", _7_sub_into_vid.merge_subtitles_to_video),
    ]


def get_dubbing_steps():
    return [
        ("🔊 Generate TTS tasks", _8_1_audio_task.gen_audio_task_main),
        ("🧩 Build dubbing chunks", _8_2_dub_chunks.gen_dub_chunks),
        ("🎵 Extract reference audio", _9_refer_audio.extract_refer_audio_main),
        ("🗣️ Generate audio", _10_gen_audio.gen_audio),
        ("🔄 Merge full dubbed audio", _11_merge_audio.merge_full_audio),
        ("🎞️ Merge dubbed audio into video", _12_dub_to_vid.merge_video_audio),
    ]


def run_steps(stage_name, steps):
    console.print(Rule(f"[bold cyan]{stage_name}[/bold cyan]"))
    total_steps = len(steps)

    for index, (label, func) in enumerate(steps, start=1):
        console.print(
            Panel(
                f"[bold green]{label}[/]",
                title=f"{stage_name} | step {index}/{total_steps}",
                border_style="blue",
            )
        )
        try:
            func()
        except Exception as exc:
            console.print(
                Panel(
                    f"[bold red]Stage failed:[/]\n{label}\n\n{exc}",
                    title="Error",
                    border_style="red",
                )
            )
            raise

    console.print(
        Panel(
            f"[bold green]{stage_name} completed successfully[/]",
            border_style="green",
        )
    )


def main():
    console.print(
        Panel(
            "[bold magenta]VideoLingo CLI pipeline started[/]",
            border_style="magenta",
        )
    )
    run_steps("Subtitle Pipeline", get_translate_steps())
    run_steps("Dubbing Pipeline", get_dubbing_steps())
    console.print(
        Panel(
            "[bold green]All stages completed successfully[/]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
