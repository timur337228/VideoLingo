import json
from core.utils import *

## ================================================================
# @ step4_splitbymeaning.py
def get_split_prompt(sentence, num_parts = 2, word_limit = 20):
    language = load_key("whisper.detected_language")
    split_prompt = f"""
## Role
You are a professional Netflix subtitle splitter in **{language}**.

## Task
Split the given subtitle text into **{num_parts}** parts, each less than **{word_limit}** words.

1. Maintain sentence meaning coherence according to Netflix subtitle standards
2. MOST IMPORTANT: Keep parts roughly equal in length (minimum 3 words each)
3. Split at natural points like punctuation marks or conjunctions
4. If provided text is repeated words, simply split at the middle of the repeated words.

## Steps
1. Analyze the sentence structure, complexity, and key splitting challenges
2. Generate two alternative splitting approaches with [br] tags at split positions
3. Compare both approaches highlighting their strengths and weaknesses
4. Choose the best splitting approach

## Given Text
<split_this_sentence>
{sentence}
</split_this_sentence>

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief description of sentence structure, complexity, and key splitting challenges",
    "split1": "First splitting approach with [br] tags at split positions",
    "split2": "Alternative splitting approach with [br] tags at split positions",
    "assess": "Comparison of both approaches highlighting their strengths and weaknesses",
    "choice": "1 or 2"
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
""".strip()
    return split_prompt

"""{{
    "analysis": "Brief analysis of the text structure",
    "split": "Complete sentence with [br] tags at split positions"
}}"""

## ================================================================
# @ step4_1_summarize.py
def get_summary_prompt(source_content, custom_terms_json=None):
    src_lang = load_key("whisper.detected_language")
    tgt_lang = load_key("target_language")
    
    # add custom terms note
    terms_note = ""
    if custom_terms_json:
        terms_list = []
        for term in custom_terms_json['terms']:
            terms_list.append(f"- {term['src']}: {term['tgt']} ({term['note']})")
        terms_note = "\n### Existing Terms\nPlease exclude these terms in your extraction:\n" + "\n".join(terms_list)
    
    summary_prompt = f"""
## Role
You are a video translation expert and terminology consultant, specializing in {src_lang} comprehension and {tgt_lang} expression optimization.

## Task
For the provided {src_lang} video text:
1. Summarize main topic in two sentences
2. Extract professional terms/names with {tgt_lang} translations (excluding existing terms)
3. Provide brief explanation for each term

{terms_note}

Steps:
1. Topic Summary:
   - Quick scan for general understanding
   - Write two sentences: first for main topic, second for key point
2. Term Extraction:
   - Mark professional terms and names (excluding those listed in Existing Terms)
   - Provide {tgt_lang} translation or keep original
   - Add brief explanation
   - Extract less than 15 terms

## INPUT
<text>
{source_content}
</text>

## Output in only JSON format and no other text
{{
  "theme": "Two-sentence video summary",
  "terms": [
    {{
      "src": "{src_lang} term",
      "tgt": "{tgt_lang} translation or original", 
      "note": "Brief explanation"
    }},
    ...
  ]
}}  

## Example
{{
  "theme": "本视频介绍人工智能在医疗领域的应用现状。重点展示了AI在医学影像诊断和药物研发中的突破性进展。",
  "terms": [
    {{
      "src": "Machine Learning",
      "tgt": "机器学习",
      "note": "AI的核心技术，通过数据训练实现智能决策"
    }},
    {{
      "src": "CNN",
      "tgt": "CNN",
      "note": "卷积神经网络，用于医学图像识别的深度学习模型"
    }}
  ]
}}

Note: Start you answer with ```json and end with ```, do not add any other text.
""".strip()
    return summary_prompt

## ================================================================
# @ step5_translate.py & translate_lines.py
def generate_shared_prompt(previous_content_prompt, after_content_prompt, summary_prompt, things_to_note_prompt):
    return f'''### Context Information
<previous_content>
{previous_content_prompt}
</previous_content>

<subsequent_content>
{after_content_prompt}
</subsequent_content>

### Content Summary
{summary_prompt}

### Points to Note
{things_to_note_prompt}'''

def get_prompt_faithfulness(lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    # Split lines by \n
    line_splits = lines.split('\n')
    json_dict = {}
    for i, line in enumerate(line_splits, 1):
        json_dict[f"{i}"] = {"origin": line, "direct": f"direct {TARGET_LANGUAGE} translation {i}."}
    json_format = json.dumps(json_dict, indent=2, ensure_ascii=False)

    src_language = load_key("whisper.detected_language")
    prompt_faithfulness = f'''
## Role
You are a professional Netflix subtitle translator, fluent in both {src_language} and {TARGET_LANGUAGE}, as well as their respective cultures. 
Your expertise lies in accurately understanding the semantics and structure of the original {src_language} text and faithfully translating it into {TARGET_LANGUAGE} while preserving the original meaning.

## Task
We have a segment of original {src_language} subtitles that need to be directly translated into {TARGET_LANGUAGE}. These subtitles come from a specific context and may contain specific themes and terminology.

1. Translate the original {src_language} subtitles into {TARGET_LANGUAGE} line by line
2. Ensure the translation is faithful to the original, accurately conveying the original meaning
3. Consider the context and professional terminology

{shared_prompt}

<translation_principles>
1. Faithful to the original: Accurately convey the content and meaning of the original text, without arbitrarily changing, adding, or omitting content.
2. Accurate terminology: Use professional terms correctly and maintain consistency in terminology.
3. Understand the context: Fully comprehend and reflect the background and contextual relationships of the text.
</translation_principles>

## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format and no other text
```json
{json_format}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''
    return prompt_faithfulness.strip()


def get_prompt_fast_translation(lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    src_language = load_key("whisper.detected_language")
    language_code = load_key("language_code")
    line_splits = lines.split('\n')
    json_dict = {}
    for i, line in enumerate(line_splits, 1):
        json_dict[f"{i}"] = {
            "origin": line,
            "direct": f"faithful {TARGET_LANGUAGE} translation {i}",
            "reflect": "OK or brief wording/meaning note",
            "final": f"natural, meaning-preserving {TARGET_LANGUAGE} subtitle {i}"
        }
    json_format = json.dumps(json_dict, indent=2, ensure_ascii=False)

    prompt_fast_translation = f'''
## Role
You are a professional Netflix subtitle translator fluent in {src_language} and {TARGET_LANGUAGE}.

## Task
Translate the original {src_language} subtitles into natural {TARGET_LANGUAGE} subtitles in one pass.
For each line, do three compact steps internally:
1. Create `direct`: a faithful translation of the source line
2. Create `reflect`: write `OK` or a very brief note if `direct` needs polishing
3. Create `final`: a natural subtitle that preserves every meaning-bearing detail from `origin` and `direct`

{shared_prompt}

## Non-negotiable Rules
- Keep one-to-one alignment: one input line -> one output item
- Do not merge, split, repeat, or move meaning between neighboring lines
- Preserve subject identity, quantities, comparisons, degree, negation, modality, time references, and cause/effect
- Preserve qualifiers and scope markers such as average, some, many, most, only, almost, nearly, about, at least, fewer, more, still, already, no, and not
- If naturalness and meaning conflict, choose meaning
- If the target language ({language_code}) requires grammatical gender and the source or context makes gender clear, use the correct gendered forms
- If speaker gender is not clear from the source/context, keep the translation faithful and natural; the separate gender-inflection stage may adjust speaker-specific agreement later
- Do not add explanations or comments to `final`

## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format and no other text
```json
{json_format}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''
    return prompt_fast_translation.strip()


def get_prompt_expressiveness(faithfulness_result, lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    json_format = {
        key: {
            "origin": value["origin"],
            "direct": value["direct"],
            "reflect": "briefly note only fluency or wording issues; write OK if no change is needed",
            "free": "your polished but meaning-preserving translation"
        }
        for key, value in faithfulness_result.items()
    }
    json_format = json.dumps(json_format, indent=2, ensure_ascii=False)

    src_language = load_key("whisper.detected_language")
    prompt_expressiveness = f'''
## Role
You are a professional Netflix subtitle translator and language consultant.
Your expertise lies not only in accurately understanding the original {src_language} but also in optimizing the {TARGET_LANGUAGE} translation to better suit the target language's expression habits and cultural background.

## Task
We already have a direct translation version of the original {src_language} subtitles.
Your task is to polish these direct translations so they sound natural and fluent in {TARGET_LANGUAGE} without changing the meaning of the original text.

1. Analyze the direct translation line by line, but only for fluency, concision, and idiomatic wording
2. Make the smallest possible edits needed to improve readability
3. Preserve every meaning-bearing element from the original source and the direct translation
4. If the direct translation is already good, keep it unchanged
5. Do not add comments or explanations in the translation, as the subtitles are for the audience to read
6. Do not leave empty lines in the polished translation, as the subtitles are for the audience to read
7. Each output line must translate only its own input line
8. Do not merge neighboring lines into one sentence
9. Do not repeat or copy content from adjacent lines
10. If an input line is a fragment, keep it as a matching fragment instead of expanding it with neighboring meaning

{shared_prompt}

<Line Boundary Rules>
- Preserve one-to-one alignment between input and output lines
- Never use the same full translation for two different neighboring source lines unless the source lines themselves are effectively identical
- If the meaning spans multiple source lines, keep each output line limited to the meaning expressed in that specific source line
- Do not move key information from one line into another line
</Line Boundary Rules>

<Meaning Preservation Rules>
- Treat `direct` as the semantic anchor and improve wording conservatively
- Never delete or weaken qualifiers, scope markers, or specificity
- Preserve subject identity, quantities, comparisons, degree, negation, modality, time references, and cause/effect relations
- Words such as average, some, many, most, only, almost, nearly, about, at least, fewer, more, still, already, no, and not must be preserved in meaning, even if rephrased
- If meaning and naturalness conflict, preserve meaning
- Before finalizing each line, verify that the polished line still says everything important that both `origin` and `direct` say
</Meaning Preservation Rules>

<Translation Analysis Steps>
Please use a two-step thinking process to handle the text line by line:

1. Direct Translation Reflection:
   - Evaluate language fluency
   - Check if the language style is consistent with the original text
   - Check the conciseness of the subtitles, but do not remove meaning-bearing details

2. {TARGET_LANGUAGE} Natural-but-Faithful Rewrite:
   - Aim for contextual smoothness and naturalness, conforming to {TARGET_LANGUAGE} expression habits
   - Ensure it's easy for {TARGET_LANGUAGE} audience to understand and accept
   - Adapt the language style to match the theme (e.g., use casual language for tutorials, professional terminology for technical content, formal language for documentaries)
   - Keep every qualifier and factual detail that affects meaning
</Translation Analysis Steps>
   
## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format and no other text
```json
{json_format}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''
    return prompt_expressiveness.strip()


def get_prompt_meaning_preservation(faithfulness_result, expressiveness_result, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    payload = {}
    for key, value in faithfulness_result.items():
        express_item = expressiveness_result.get(key, {})
        payload[key] = {
            "origin": value["origin"],
            "direct": value["direct"],
            "free": express_item.get("free", value["direct"]),
            "final": "final subtitle after meaning check"
        }
    json_format = json.dumps(payload, indent=2, ensure_ascii=False)

    src_language = load_key("whisper.detected_language")
    prompt_meaning_preservation = f'''
## Role
You are a senior subtitle reviewer responsible for protecting meaning while keeping subtitles natural.

## Task
You will receive, for each subtitle line:
- `origin`: the original {src_language} line
- `direct`: a faithful direct translation into {TARGET_LANGUAGE}
- `free`: a polished version that may be more natural

Your job is to produce `final`, the best subtitle line for the audience.

1. If `free` is natural and fully faithful to both `origin` and `direct`, copy it unchanged into `final`
2. If `free` drops, weakens, or changes any meaning, repair it with the smallest possible edit
3. Use `direct` as the semantic anchor whenever `free` becomes too loose
4. When meaning and style conflict, choose meaning
5. Keep one-to-one line alignment and never merge or split lines

{shared_prompt}

<Meaning Preservation Checklist>
- Preserve qualifiers and scope: average, some, many, most, only, almost, nearly, about, at least, exactly
- Preserve comparison and degree: more, less, fewer, better, worse, first, last
- Preserve negation and modality: no, not, never, may, might, should, must
- Preserve time and factual anchors: now, already, still, before, after, numbers, dates, durations
- Preserve who did what to whom
- If a detail is present in `origin` and `direct`, it must still be present in `final`
</Meaning Preservation Checklist>

## Output in only JSON format and no other text
```json
{json_format}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''
    return prompt_meaning_preservation.strip()


## ================================================================
# @ step6_splitforsub.py
def get_align_prompt(src_sub, tr_sub, src_part):
    targ_lang = load_key("target_language")
    src_lang = load_key("whisper.detected_language")
    src_splits = src_part.split('\n')
    num_parts = len(src_splits)
    src_part = src_part.replace('\n', ' [br] ')
    align_parts_json = ','.join(
        f'''
        {{
            "src_part_{i+1}": "{src_splits[i]}",
            "target_part_{i+1}": "Corresponding aligned {targ_lang} subtitle part"
        }}''' for i in range(num_parts)
    )

    align_prompt = f'''
## Role
You are a Netflix subtitle alignment expert fluent in both {src_lang} and {targ_lang}.

## Task
We have {src_lang} and {targ_lang} original subtitles for a Netflix program, as well as a pre-processed split version of {src_lang} subtitles.
Your task is to create the best splitting scheme for the {targ_lang} subtitles based on this information.

1. Analyze the word order and structural correspondence between {src_lang} and {targ_lang} subtitles
2. Split the {targ_lang} subtitles according to the pre-processed {src_lang} split version
3. Never leave empty lines. If it's difficult to split based on meaning, you may appropriately rewrite the sentences that need to be aligned
4. Do not add comments or explanations in the translation, as the subtitles are for the audience to read

## INPUT
<subtitles>
{src_lang} Original: "{src_sub}"
{targ_lang} Original: "{tr_sub}"
Pre-processed {src_lang} Subtitles ([br] indicates split points): {src_part}
</subtitles>

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief analysis of word order, structure, and semantic correspondence between two subtitles",
    "align": [
        {align_parts_json}
    ]
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
    return align_prompt

## ================================================================
# @ step8_gen_audio_task.py @ step10_gen_audio.py
def get_subtitle_trim_prompt(text, duration):
 
    rule = '''Consider a. Reducing filler words without modifying meaningful content. b. Omitting unnecessary modifiers or pronouns, for example:
    - "Please explain your thought process" can be shortened to "Please explain thought process"
    - "We need to carefully analyze this complex problem" can be shortened to "We need to analyze this problem"
    - "Let's discuss the various different perspectives on this topic" can be shortened to "Let's discuss different perspectives on this topic"
    - "Can you describe in detail your experience from yesterday" can be shortened to "Can you describe yesterday's experience" '''

    trim_prompt = f'''
## Role
You are a professional subtitle editor, editing and optimizing lengthy subtitles that exceed voiceover time before handing them to voice actors. 
Your expertise lies in cleverly shortening subtitles slightly while ensuring the original meaning and structure remain unchanged.

## INPUT
<subtitles>
Subtitle: "{text}"
Duration: {duration} seconds
</subtitles>

## Processing Rules
{rule}

## Processing Steps
Please follow these steps and provide the results in the JSON output:
1. Analysis: Briefly analyze the subtitle's structure, key information, and filler words that can be omitted.
2. Trimming: Based on the rules and analysis, optimize the subtitle by making it more concise according to the processing rules.

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief analysis of the subtitle, including structure, key information, and potential processing locations",
    "result": "Optimized and shortened subtitle in the original subtitle language"
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
    return trim_prompt

## ================================================================
# @ tts_main
def get_correct_text_prompt(text):
    return f'''
## Role
You are a text cleaning expert for TTS (Text-to-Speech) systems.

## Task
Clean the given text by:
1. Keep only basic punctuation (.,?!)
2. Preserve the original meaning

## INPUT
{text}

## Output in only JSON format and no other text
```json
{{
    "text": "cleaned text here"
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()


def build_gender_prompt(records, gender):
    target_language = load_key("target_language")
    language_code = load_key("language_code")
    language_specific_rules = {
        "ru": "Adjust first-person verbs, reflexive forms, short adjectives, participles, and explicit self-descriptions that refer to the speaker. Example for a female speaker: `Я хотел объяснить` -> `Я хотела объяснить`.",
        "es": "Adjust gendered self-reference, adjectives, participles, and nouns that clearly describe the speaker. Do not force gender where Spanish normally stays neutral. Example for a female speaker: `Estoy listo` -> `Estoy lista`.",
        "fr": "Adjust written agreement that refers to the speaker, such as self-descriptive adjectives and participles. Leave unchanged where French normally does not mark speaker gender. Example for a female speaker: `Je suis prêt` -> `Je suis prête`.",
    }
    input_payload = {
        str(index): {
            "source": record["source"],
            "translation": record["translation"],
        }
        for index, record in enumerate(records, start=1)
    }

    language_rule = language_specific_rules.get(
        language_code,
        "Only adjust forms that truly need speaker-gender agreement.",
    )

    return f"""
You will receive subtitle lines translated into {target_language}, together with their original source lines.

All lines in this batch belong to one speaker whose gender is: {gender}.

Your goal is to make the smallest possible edits so the existing translation agrees with that speaker gender.
The current translation may contain the wrong default gender. If so, fixing that gender is required, not optional.

Rules:
1. Use the source line only for meaning and reference disambiguation.
2. Keep every output line as close as possible to the current translation.
3. Change every word that requires speaker-gender agreement.
4. Do not rewrite style, tone, wording, or sentence structure unless a gender fix requires it.
5. If a line does not require gender marking in {target_language}, return it unchanged.
6. If the line refers to someone other than the speaker, do not change that wording to match the speaker.
7. Preserve the number of items exactly.
8. Never merge or split lines.
9. Return JSON only.

Language-specific note for {language_code}: {language_rule}

Return this format:
{{
  "1": {{"text": "line 1"}},
  "2": {{"text": "line 2"}}
}}

Input:
{json.dumps(input_payload, ensure_ascii=False, indent=2)}
""".strip()
