import os
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from core import *


console = Console()


# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_translate_steps():
    return [
        ("🎙️ WhisperX transcription", _2_asr.transcribe),
        ("Get gender speakers", _2_1_speakers_profiles.get_gender_speakers),
        ("✂️ NLP sentence split", _3_1_split_nlp.split_by_spacy),
        ("🧠 Meaning split", _3_2_split_meaning.split_sentences_by_meaning),
        ("📝 Summary and terminology", _4_1_summarize.get_summary),
        ("🌍 Translation", _4_2_translate.translate_all),
        ("Adding gender in translation", _4_3_gender_inflection.gender_inflection),
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


def print_timing_summary(timings, total_elapsed):
    table = Table(title="Pipeline timing summary", show_lines=False)
    table.add_column("Stage", style="cyan")
    table.add_column("Step", style="green")
    table.add_column("Time", justify="right", style="yellow")

    for item in timings:
        table.add_row(
            item["stage"],
            item["label"],
            format_duration(item["elapsed"]),
        )

    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        "[bold]Whole video[/bold]",
        f"[bold]{format_duration(total_elapsed)}[/bold]",
    )
    console.print(table)


def run_steps(stage_name, steps):
    console.print(Rule(f"[bold cyan]{stage_name}[/bold cyan]"))
    total_steps = len(steps)
    stage_timings = []
    stage_started_at = time.perf_counter()

    for index, (label, func) in enumerate(steps, start=1):
        console.print(
            Panel(
                f"[bold green]{label}[/]",
                title=f"{stage_name} | step {index}/{total_steps}",
                border_style="blue",
            )
        )
        step_started_at = time.perf_counter()
        try:
            func()
        except Exception as exc:
            elapsed = time.perf_counter() - step_started_at
            console.print(
                Panel(
                    (
                        f"[bold red]Stage failed:[/]\n{label}\n\n"
                        f"Elapsed: {format_duration(elapsed)}\n\n{exc}"
                    ),
                    title="Error",
                    border_style="red",
                )
            )
            raise
        elapsed = time.perf_counter() - step_started_at
        stage_timings.append(
            {
                "stage": stage_name,
                "label": label,
                "elapsed": elapsed,
            }
        )
        console.print(
            Panel(
                (
                    f"[bold green]{label} completed[/]\n"
                    f"Elapsed: [bold yellow]{format_duration(elapsed)}[/]"
                ),
                title=f"{stage_name} | step {index}/{total_steps}",
                border_style="green",
            )
        )

    stage_elapsed = time.perf_counter() - stage_started_at
    console.print(
        Panel(
            (
                f"[bold green]{stage_name} completed successfully[/]\n"
                f"Stage time: [bold yellow]{format_duration(stage_elapsed)}[/]"
            ),
            border_style="green",
        )
    )
    return stage_timings


def main():
    pipeline_started_at = time.perf_counter()
    timings = []
    console.print(
        Panel(
            "[bold magenta]VideoLingo CLI pipeline started[/]",
            border_style="magenta",
        )
    )
    timings.extend(run_steps("Subtitle Pipeline", get_translate_steps()))
    if not load_key("get_only_sub_video"):
        timings.extend(run_steps("Dubbing Pipeline", get_dubbing_steps()))
    total_elapsed = time.perf_counter() - pipeline_started_at
    console.print(
        Panel(
            (
                "[bold green]All stages completed successfully[/]\n"
                f"Total video time: [bold yellow]{format_duration(total_elapsed)}[/]"
            ),
            border_style="green",
        )
    )
    print_timing_summary(timings, total_elapsed)


if __name__ == "__main__":
    main()
