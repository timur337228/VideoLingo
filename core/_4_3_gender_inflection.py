import pandas as pd
import json
from core.prompts import build_gender_prompt
from core._4_2_translate import split_chunks_by_chars
from core.translate_lines import translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
from core._8_1_audio_task import check_len_then_trim
from core._6_gen_sub import align_timestamp
from core.utils import *
from core.utils.models import *


def split_records_into_chunks(records, chunk_size=600, max_i=10):
    chunks = []
    current_indices = []
    current_lines = []
    current_len = 0

    for record in records:
        line = str(record["text"]).strip()
        add_len = len(line) + 1

        if current_lines and (current_len + add_len > chunk_size or len(current_lines) >= max_i):
            chunks.append({
                "indices": current_indices,
                "text": "\n".join(current_lines),
            })
            current_indices = []
            current_lines = []
            current_len = 0

        current_indices.append(record["idx"])
        current_lines.append(line)
        current_len += add_len

    if current_lines:
        chunks.append({
            "indices": current_indices,
            "text": "\n".join(current_lines),
        })

    return chunks


def gender_inflection():
    if not (load_key("is_gender_translate") and load_key("whisper.enable_diarization")):
        return
    df = pd.read_csv(_4_2_TRANSLATION)
    df_text = pd.read_csv(_2_CLEANED_CHUNKS)
    df_text["text"] = df_text["text"].str.strip("").str.strip()
    genders = load_key("genders_speakers")
    for speaker_id, group_df in df.groupby("speaker_id", sort=False):
        gender = genders.get(speaker_id)
        if not gender:
            continue
        records = [
            {"idx": idx, "text": row["Translation"]}
            for idx, row in group_df.iterrows()
        ]

        chunks = split_records_into_chunks(records)
        for chunk in chunks:
            response = ask_gpt(
                build_gender_prompt(chunk["text"], gender),
                resp_type="json",
                log_title="gender_inflection"
            )

            new_lines = [
                response[str(i)]["text"].strip()
                for i in range(1, len(chunk["indices"]) + 1)
            ]
            if len(new_lines) != len(chunk["indices"]):
                rprint("[red]LLM returned wrong number of lines[/red]")
                continue
                # raise ValueError("LLM returned wrong number of lines")
            for idx, new_line in zip(chunk["indices"], new_lines):
                df.at[idx, "Translation"] = new_line
    df = df.drop(columns=["timestamp", "duration", "speaker_id"], errors="ignore")
    df.to_csv(_4_2_TRANSLATION, index=False)