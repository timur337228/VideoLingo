import pandas as pd
from rich.panel import Panel

from core._6_gen_sub import align_timestamp
from core.prompts import build_gender_prompt
from core.utils import *
from core.utils.models import *


MAX_GENDER_ATTEMPTS = 2


def _clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_records_into_chunks(records, chunk_size=900, max_i=8):
    chunks = []
    current_records = []
    current_len = 0

    for record in records:
        add_len = len(record["source"]) + len(record["translation"]) + 16

        if current_records and (current_len + add_len > chunk_size or len(current_records) >= max_i):
            chunks.append(current_records)
            current_records = []
            current_len = 0

        current_records.append(record)
        current_len += add_len

    if current_records:
        chunks.append(current_records)

    return chunks


def _validate_gender_result(response_data, records):
    if not isinstance(response_data, dict):
        return {"status": "error", "message": "Gender response is not a JSON object"}

    expected_keys = [str(i) for i in range(1, len(records) + 1)]
    missing_keys = [key for key in expected_keys if key not in response_data]
    if missing_keys:
        return {"status": "error", "message": f"Missing required key(s): {', '.join(missing_keys)}"}

    for index, record in enumerate(records, start=1):
        item = response_data.get(str(index))
        if not isinstance(item, dict):
            return {"status": "error", "message": f"Item {index} must be an object"}

        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"status": "error", "message": f"Item {index} must contain a non-empty text field"}

    return {"status": "success", "message": "success"}


def _best_effort_gender_result(response_data, records):
    response_data = response_data if isinstance(response_data, dict) else {}
    normalized = {}

    for index, record in enumerate(records, start=1):
        item = response_data.get(str(index), {})
        text = item.get("text") if isinstance(item, dict) else None
        cleaned_text = _clean_text(text).replace("\n", " ")
        normalized[str(index)] = {"text": cleaned_text or record["translation"]}

    return normalized


def _apply_gender_chunk(records, gender):
    prompt = build_gender_prompt(records, gender)
    last_result = None
    last_error = None

    for retry in range(MAX_GENDER_ATTEMPTS):
        try:
            result = ask_gpt(prompt + retry * " ", resp_type="json", log_title="gender_inflection")
            last_result = result
            valid_result = _validate_gender_result(result, records)
            if valid_result["status"] == "success":
                return _best_effort_gender_result(result, records)
            last_error = valid_result["message"]
        except Exception as exc:
            last_error = str(exc)

        if retry != MAX_GENDER_ATTEMPTS - 1:
            rprint("[yellow]⚠️ Gender inflection chunk failed validation, retrying...[/yellow]")

    if last_error:
        rprint(f"[yellow]⚠️ Gender inflection is using best-effort fallback: {last_error}[/yellow]")

    return _best_effort_gender_result(last_result, records)


def _ensure_translation_metadata(df, df_text):
    required_columns = {"speaker_id", "timestamp", "duration"}
    if required_columns.issubset(df.columns):
        return df

    if not {"Source", "Translation"}.issubset(df.columns):
        missing = sorted({"Source", "Translation"} - set(df.columns))
        rprint(
            Panel(
                f"Skipping gender inflection because translation metadata is missing and "
                f"the file also lacks columns needed for recovery: {', '.join(missing)}",
                title="Warning",
                border_style="yellow",
            )
        )
        return df

    rprint(
        Panel(
            "translation_results.csv is missing speaker metadata. Rebuilding speaker_id and "
            "timestamps from cleaned_chunks.csv before gender inflection.",
            title="Info",
            border_style="yellow",
        )
    )

    try:
        rebuilt = align_timestamp(
            df_text,
            df[["Source", "Translation"]].copy(),
            subtitle_output_configs=[],
            output_dir=None,
            for_display=False,
        )
    except Exception as exc:
        rprint(
            Panel(
                f"Unable to rebuild speaker metadata for gender inflection: {exc}",
                title="Warning",
                border_style="yellow",
            )
        )
        return df

    for column in df.columns:
        if column not in rebuilt.columns:
            rebuilt[column] = df[column]

    return rebuilt


def gender_inflection():
    if not (load_key("is_gender_translate") and load_key("whisper.enable_diarization")):
        return

    df = pd.read_csv(_4_2_TRANSLATION)
    df_text = pd.read_csv(_2_CLEANED_CHUNKS)
    df_text["text"] = df_text["text"].astype(str).str.strip('"').str.strip()

    df = _ensure_translation_metadata(df, df_text)
    if "speaker_id" not in df.columns:
        rprint(Panel("No speaker metadata available, skipping gender inflection.", title="Info", border_style="yellow"))
        return

    genders = load_key("genders_speakers") or {}
    if not isinstance(genders, dict) or not genders:
        rprint(Panel("No speaker genders available, skipping gender inflection.", title="Info", border_style="yellow"))
        return

    changed_lines = 0
    processed_speakers = 0

    for speaker_id, group_df in df.groupby("speaker_id", sort=False):
        if pd.isna(speaker_id):
            continue

        gender = genders.get(speaker_id)
        if not gender:
            continue

        records = []
        for idx, row in group_df.iterrows():
            translation = _clean_text(row.get("Translation"))
            if not translation:
                continue

            records.append(
                {
                    "idx": idx,
                    "source": _clean_text(row.get("Source")),
                    "translation": translation,
                }
            )

        if not records:
            continue

        processed_speakers += 1
        for chunk_records in split_records_into_chunks(records):
            response = _apply_gender_chunk(chunk_records, gender)

            for line_index, record in enumerate(chunk_records, start=1):
                new_line = response[str(line_index)]["text"].strip()
                if df.at[record["idx"], "Translation"] != new_line:
                    changed_lines += 1
                    df.at[record["idx"], "Translation"] = new_line

    df.to_csv(_4_2_TRANSLATION, index=False)
    rprint(
        Panel(
            f"Gender inflection updated {changed_lines} subtitle line(s) across {processed_speakers} speaker(s).",
            title="Success",
            border_style="green",
        )
    )
