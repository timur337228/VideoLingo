from core.prompts import generate_shared_prompt, get_prompt_faithfulness, get_prompt_expressiveness
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from difflib import SequenceMatcher
from core.utils import *
console = Console()
MAX_TRANSLATION_ATTEMPTS = 2

def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _best_effort_translation_result(response_data, source_lines, value_key, fallback_items=None):
    response_data = response_data if isinstance(response_data, dict) else {}
    fallback_items = fallback_items if isinstance(fallback_items, dict) else {}
    normalized = {}

    for idx, source_line in enumerate(source_lines, start=1):
        key = str(idx)
        item = response_data.get(key, {})
        if not isinstance(item, dict):
            item = {}

        fallback_item = fallback_items.get(key, {})
        if not isinstance(fallback_item, dict):
            fallback_item = {}

        origin = item.get("origin") or fallback_item.get("origin") or source_line
        value = item.get(value_key)
        if not isinstance(value, str) or not value.strip():
            value = fallback_item.get(value_key)

        if not isinstance(value, str) or not value.strip():
            if value_key == "free":
                value = fallback_item.get("direct") or source_line
            else:
                value = source_line

        normalized[key] = {
            "origin": origin,
            value_key: value.replace('\n', ' ').strip(),
        }

    return normalized

def valid_express_alignment(response_data):
    keys = sorted(response_data.keys(), key=int)
    for idx in range(len(keys) - 1):
        current = response_data[keys[idx]]
        following = response_data[keys[idx + 1]]

        current_free = current["free"].strip()
        next_free = following["free"].strip()
        current_origin = current["origin"].strip()
        next_origin = following["origin"].strip()

        if not current_free or not next_free:
            continue

        free_similarity = _text_similarity(current_free, next_free)
        origin_similarity = _text_similarity(current_origin, next_origin)
        long_enough = min(len(current_free), len(next_free)) >= 18

        # Catch the common failure mode where the model merges neighboring source lines
        # into one natural sentence and copies it into both adjacent outputs.
        if free_similarity >= 0.97 and origin_similarity <= 0.75 and long_enough:
            return {
                "status": "error",
                "message": f"Suspicious duplicated neighboring free translations at lines {keys[idx]} and {keys[idx + 1]}",
            }
    return {"status": "success", "message": "success validate"}


def valid_translate_result(result: dict, required_keys: list, required_sub_keys: list):
    # Check for the required key
    if not all(key in result for key in required_keys):
        return {"status": "error", "message": f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}"}
    
    # Check for required sub-keys in all items
    for key in result:
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            return {"status": "error", "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}"}

    return {"status": "success", "message": "Translation completed"}

def translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt, index = 0, gender = None):
    shared_prompt = generate_shared_prompt(previous_content_prompt, after_cotent_prompt, summary_prompt, things_to_note_prompt, gender)
    
    # Retry translation if the length of the original text and the translated text are not the same, or if the specified key is missing
    def retry_translation(prompt, length, step_name, fallback_items=None):
        def valid_faith(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length+1)], ['direct'])
        def valid_express(response_data):
            base_check = valid_translate_result(response_data, [str(i) for i in range(1, length+1)], ['free'])
            if base_check["status"] != "success":
                return base_check
            return valid_express_alignment(response_data)

        source_lines = lines.split('\n')
        validator = valid_faith if step_name == 'faithfulness' else valid_express
        value_key = 'direct' if step_name == 'faithfulness' else 'free'
        last_result = None
        last_error = None

        for retry in range(MAX_TRANSLATION_ATTEMPTS):
            try:
                result = ask_gpt(prompt + retry * " ", resp_type='json', log_title=f'translate_{step_name}')
                last_result = result
                valid_result = validator(result)
                if valid_result["status"] == "success" and isinstance(result, dict) and len(source_lines) == len(result):
                    return result
                last_error = valid_result["message"]
            except Exception as e:
                last_error = str(e)

            if retry != MAX_TRANSLATION_ATTEMPTS - 1:
                console.print(f'[yellow]⚠️ {step_name.capitalize()} translation of block {index} failed, Retry...[/yellow]')

        console.print(
            f'[yellow]⚠️ {step_name.capitalize()} translation of block {index} failed after '
            f'{MAX_TRANSLATION_ATTEMPTS} attempts. Using best-effort result and continuing...[/yellow]'
        )
        if last_error:
            console.print(f'[yellow]↳ Last validation error: {last_error}[/yellow]')

        return _best_effort_translation_result(last_result, source_lines, value_key, fallback_items)

    ## Step 1: Faithful to the Original Text
    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, len(lines.split('\n')), 'faithfulness')

    for i in faith_result:
        faith_result[i]["direct"] = faith_result[i]["direct"].replace('\n', ' ')

    # If reflect_translate is False or not set, use faithful translation directly
    reflect_translate = load_key('reflect_translate')
    if not reflect_translate:
        # If reflect_translate is False or not set, use faithful translation directly
        translate_result = "\n".join([faith_result[i]["direct"].strip() for i in faith_result])
        
        table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
        table.add_column("Translations", style="bold")
        for i, key in enumerate(faith_result):
            table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
            table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
            if i < len(faith_result) - 1:
                table.add_row("[yellow]" + "-" * 50 + "[/yellow]")
        
        console.print(table)
        return translate_result, lines

    ## Step 2: Express Smoothly  
    prompt2 = get_prompt_expressiveness(faith_result, lines, shared_prompt)
    express_result = retry_translation(prompt2, len(lines.split('\n')), 'expressiveness', fallback_items=faith_result)

    table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
    table.add_column("Translations", style="bold")
    for i, key in enumerate(express_result):
        table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
        table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
        table.add_row(f"[green]Free:    {express_result[key]['free']}[/green]")
        if i < len(express_result) - 1:
            table.add_row("[yellow]" + "-" * 50 + "[/yellow]")

    console.print(table)

    translate_result = "\n".join([express_result[i]["free"].replace('\n', ' ').strip() for i in express_result])

    if len(lines.split('\n')) != len(translate_result.split('\n')):
        console.print(Panel(f'[red]❌ Translation of block {index} failed, Length Mismatch, Please check `output/gpt_log/translate_expressiveness.json`[/red]'))
        raise ValueError(f'Origin ···{lines}···,\nbut got ···{translate_result}···')

    return translate_result, lines


if __name__ == '__main__':
    # test e.g.
    lines = '''All of you know Andrew Ng as a famous computer science professor at Stanford.
He was really early on in the development of neural networks with GPUs.
Of course, a creator of Coursera and popular courses like deeplearning.ai.
Also the founder and creator and early lead of Google Brain.'''
    previous_content_prompt = None
    after_cotent_prompt = None
    things_to_note_prompt = None
    summary_prompt = None
    translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt)
