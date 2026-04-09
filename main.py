import streamlit as st
import os, sys, time
from core.st_utils.imports_and_utils import *
from core.st_utils.task_runner import TaskRunner
from core import *

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="VideoLingo", page_icon="docs/logo.svg")

SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"

def translate():
    _2_asr.transcribe()
    _3_1_split_nlp.split_by_spacy()
    _3_2_split_meaning.split_sentences_by_meaning()
    _4_1_summarize.get_summary()
    _4_2_translate.translate_all()
    _5_split_sub.split_for_sub_main()
    _6_gen_sub.align_timestamp_main()
    _7_sub_into_vid.merge_subtitles_to_video()


def dubbling():
    _8_1_audio_task.gen_audio_task_main()
    _8_2_dub_chunks.gen_dub_chunks()
    _9_refer_audio.extract_refer_audio_main()
    _10_gen_audio.gen_audio()
    _11_merge_audio.merge_full_audio()
    _12_dub_to_vid.merge_video_audio()



def main():
    translate()
    dubbling()

if __name__ == "__main__":
    main()
