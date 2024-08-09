import math
import os
import gc
import re
import sys
import time
import glob
import json
import site
import datetime

import torch
import signal
import random
import shutil
import psutil
import string
import tempfile
import platform
import argparse
import torchaudio
import traceback
import gradio as gr
import pandas as pd
from word2number import w2n
from tqdm import tqdm
from pathlib import Path
from faster_whisper import WhisperModel  
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from metrics_logger import MetricsLogger
# Use a local Tokenizer to resolve Japanese support
# from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners
from system.ft_tokenizer.tokenizer import multilingual_cleaners
import importlib.metadata as metadata
from packaging import version
from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace

# STARTUP VARIABLES 
this_dir = Path(__file__).parent.resolve()
audio_folder = this_dir / "finetune" / "put-voice-samples-in-here"
default_path = this_dir / "finetune" / "tmp-trn"
out_path = default_path

progress = 0
theme = gr.themes.Default()
refresh_symbol = '🔄'
os.environ['TRAINER_TELEMETRY'] = '0'
pfc_status = "pass"
base_path = this_dir / "models" / "xtts"

# Set the Gradio temporary directory
gradio_temp_dir = this_dir / "finetune" / "gradio_temp"
os.environ["GRADIO_TEMP_DIR"] = str(gradio_temp_dir)

# Set validation globals to nothing
validate_train_metadata_path = None
validate_eval_metadata_path = None
validate_audio_folder = None
validate_whisper_model = None
validate_target_language = None

########################
#### Find All Model ####
########################
base_model_detected = False
def scan_models_folder():
    global base_model_detected
    models_folder = base_path
    available_models = {}
    required_files = ["config.json", "model.pth", "mel_stats.pth", "speakers_xtts.pth", "vocab.json", "dvae.pth"]
    try:
        for subfolder in models_folder.iterdir():
            if subfolder.is_dir():
                model_name = subfolder.name
                if all(subfolder.joinpath(file).exists() for file in required_files):
                    available_models[model_name] = True
                    base_model_detected = True
                else:
                    print(f"[FINETUNE] \033[91mWarning\033[0m: Model folder '{model_name}' is missing required files")
        if not available_models:
            available_models["No Model Available"] = False
            base_model_detected = False
    except FileNotFoundError:
        print("\n[FINETUNE] \033[91mError\033[0m: No XTTS models folder found. You have not yet downloaded any models or no XTTS")
        print("[FINETUNE] \033[91mError\033[0m: models can be found. Please run AllTalk and download an XTTS model that can be")
        print("[FINETUNE] \033[91mError\033[0m: used for training. Or place a full model in the following location")
        print("[FINETUNE] \033[91mError\033[0m: \033[93m\\models\\xtts\\{modelfolderhere}\033[0m")
        sys.exit(1)  # Exit the script with an error status

    return available_models

# Get available models
available_models = scan_models_folder()

#######################
#### DIAGS for PFC ####
#######################

def check_disk_space():
    global pfc_status
    # Get the current working directory
    current_directory = os.getcwd()
    # Get the disk usage statistics for the current directory's disk
    disk_usage = shutil.disk_usage(current_directory)
    # Convert the free space to GB (1GB = 1 << 30 bytes)
    free_space_gb = disk_usage.free / (1 << 30)
    # Check if the free space is more than 18GB
    is_more_than_18gb = free_space_gb > 18
    disk_space_icon = "✅"
    if not is_more_than_18gb:
        disk_space_icon ="❌"
        pfc_status = "fail"  # Update global status if disk space check fails
    # Generating the markdown text for disk space check
    disk_space_markdown = f"""
    ### 🟩 <u>Disk Space Check</u>
    &nbsp;&nbsp;&nbsp;&nbsp; {disk_space_icon} **Disk Space (> 18 GB):** {'' if is_more_than_18gb else 'You have less than 18GB on this disk '} {free_space_gb:.2f} GB
    """
    return disk_space_markdown

def test_cuda():
    global pfc_status
    cuda_home = os.environ.get('CUDA_HOME', 'N/A')
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            # Attempt to create a tensor on GPU
            torch.tensor([1.0, 2.0]).cuda()
            cuda_status = "CUDA is available and working."
            cuda_icon = "✅"
        except Exception as e:
            cuda_status = f"CUDA is available but not working. Error: {e}"
            cuda_icon = "❌"
            pfc_status = "fail"  # Update global status
    else:
        cuda_status = "CUDA is not available."
        pfc_status = "fail"  # Update global status
    return cuda_status, cuda_icon, cuda_home 

def find_files_in_path_with_wildcard(pattern):
    # Get the site-packages directory of the current Python environment
    site_packages_path = site.getsitepackages()
    found_paths = []
    # Adjust the sub-directory based on the operating system
    sub_directory = "nvidia/cublas"
    if platform.system() == "Linux":
        sub_directory = os.path.join(sub_directory, "lib")
    else:
        sub_directory = os.path.join(sub_directory, "bin")
    # Iterate over each site-packages directory (there can be more than one)
    for directory in site_packages_path:
        # Construct the search directory path
        search_directory = os.path.join(directory, sub_directory)
        # Use glob to find all files matching the pattern in this directory
        for file_path in glob.glob(os.path.join(search_directory, pattern)):
            if os.path.isfile(file_path):  # Ensure it's a file
                found_paths.append(file_path)
    return found_paths


def test_cuda():
    if torch.cuda.is_available():
        pytorch_cuda_version = torch.version.cuda
        return f'CUDA {pytorch_cuda_version} is available in this Python environment', '✅', torch.cuda.get_device_name(0)
    else:
        return 'CUDA is not available in this Python environment', '❌', 'None'

def generate_cuda_markdown():
    global pfc_status
    pfc_status = "pass"  # Initialize global status as pass

    # Check CUDA availability and get CUDA home path
    cuda_status, cuda_icon, cuda_home = test_cuda()

    # Check for specific CUDA library files
    file_name = 'cublas64_11.*' if platform.system() == "Windows" else 'libcublas.so.11*'
    found_paths = find_files_in_path_with_wildcard(file_name)
    if found_paths:
        found_paths_str = ' '.join(str(path) for path in found_paths)
        found_path_icon = '✅'
    else:
        found_paths_str = "cublas64_11 is not accessible."
        found_path_icon = '❌'
        pfc_status = "fail"  # Update global status

    # Check PyTorch version and CUDA version
    pytorch_version = torch.__version__
    pytorch_cuda_version = torch.version.cuda

    if pytorch_cuda_version in ['11.8', '12.1']:
        pytorch_cuda_version_status = ''
        pytorch_icon = '✅'
    else:
        pytorch_cuda_version_status = 'Pytorch CUDA version problem '
        pytorch_icon = '❌'
        pfc_status = "fail"  # Update global status

    cuda_markdown = f"""
    ### 🟨 <u>CUDA Information</u><br>
    &nbsp;&nbsp;&nbsp;&nbsp; {found_path_icon} **Cublas64_11 found:** {found_paths_str}  
    &nbsp;&nbsp;&nbsp;&nbsp; {pytorch_icon} **CUDA_HOME path:** {cuda_home} **(Will list your GPU if no specifc path set, which is ok)**
    """
    pytorch_markdown = f"""
    ### 🟦 <u>Python & Pytorch Information</u>  
    &nbsp;&nbsp;&nbsp;&nbsp; {pytorch_icon} **PyTorch Version:** {pytorch_cuda_version_status} {torch.__version__}  
    &nbsp;&nbsp;&nbsp;&nbsp; {cuda_icon} **CUDA is working:** {cuda_status}
    """
    return cuda_markdown, pytorch_markdown

def get_system_ram_markdown():
    global pfc_status
    virtual_memory = psutil.virtual_memory()
    total_ram_gb = virtual_memory.total / (1024 ** 3)
    available_ram_gb = virtual_memory.available / (1024 ** 3)
    used_ram_percentage = virtual_memory.percent

    # Check if the available RAM is less than 8GB
    warning_if_low_ram = available_ram_gb < 8

    # Decide the message based on the available RAM
    ram_status_message = "Warning" if warning_if_low_ram else ""
    ram_status_icon = "⚠️" if warning_if_low_ram else "✅"

    if torch.cuda.is_available():
        gpu_device_id = torch.cuda.current_device()
        gpu_device_name = torch.cuda.get_device_name(gpu_device_id) 
        # Get the total and available memory in bytes, then convert to GB
        gpu_total_mem_gb = torch.cuda.get_device_properties(gpu_device_id).total_memory / (1024 ** 3)
        # gpu_available_mem_gb = (torch.cuda.get_device_properties(gpu_device_id).total_memory - torch.cuda.memory_allocated(gpu_device_id)) / (1024 ** 3)
        # gpu_available_mem_gb = (torch.cuda.get_device_properties(gpu_device_id).total_memory - torch.cuda.memory_reserved(gpu_device_id)) / (1024 ** 3)
        gpu_reserved_mem_gb = torch.cuda.memory_reserved(gpu_device_id) / (1024 ** 3)
        gpu_available_mem_gb = gpu_total_mem_gb - gpu_reserved_mem_gb
        # Check if total or available memory is less than 11 GB and set icons
        gpu_total_status_icon = "⚠️" if gpu_total_mem_gb < 12 else "✅"
        gpu_available_status_icon = "⚠️" if gpu_available_mem_gb < 12 else "✅"
        gpu_status_icon = "✅"
    else:
        gpu_status_icon = "⚠️"
        gpu_device_name = "Cannot detect a CUDA card"
        gpu_total_mem_gb = "Cannot detect a CUDA card"
        gpu_available_mem_gb = "Cannot detect a CUDA card"
        gpu_total_status_icon = gpu_status_icon
        gpu_available_status_icon = gpu_status_icon

    system_ram_markdown = f"""
    ### 🟪 <u>System RAM and VRAM Information</u>  <br>
    &nbsp;&nbsp;&nbsp;&nbsp; {ram_status_icon} **Total RAM:** {total_ram_gb:.2f} GB<br>
    &nbsp;&nbsp;&nbsp;&nbsp; {ram_status_icon} **Available RAM:** {ram_status_message + ' - Available RAM is less than 8 GB. You have ' if warning_if_low_ram else ''} {available_ram_gb:.2f} GB available ({used_ram_percentage:.2f}% used)<br><br>
    &nbsp;&nbsp;&nbsp;&nbsp; {gpu_status_icon} **GPU Name:** {gpu_device_name}<br>
    &nbsp;&nbsp;&nbsp;&nbsp; {gpu_total_status_icon} **GPU Total RAM:** {gpu_total_mem_gb:.2f} GB<br>
    &nbsp;&nbsp;&nbsp;&nbsp; {gpu_available_status_icon} **GPU Available RAM:** {gpu_available_mem_gb:.2f} GB<br>
    """
    return system_ram_markdown

def generate_base_model_markdown(base_model_detected):
    global pfc_status
    base_model_status = 'Base model detected' if base_model_detected else 'Base model not detected'
    base_model_icon = '✅' if base_model_detected else '❌'
    base_model_markdown = f"""
    ### ⬛ <u>XTTS Base Model Detection</u>
    &nbsp;&nbsp;&nbsp;&nbsp; {base_model_icon} **Base XTTS Model Status:** {base_model_status}
    """
    return base_model_markdown

def check_tts_version(required_version="0.24.0"):
    global pfc_status
    try:
        # Get the installed version of TTS
        installed_version = metadata.version("coqui-tts")
        # Check if the installed version meets the required version
        if version.parse(installed_version) >= version.parse(required_version):
            tts_status = f"TTS version {installed_version} is installed and meets the requirement."
            tts_status_icon = "✅"
        else:
            tts_status = f"❌ Fail - TTS version {installed_version} is installed but does not meet the required version {required_version}."
            tts_status_icon = "❌"
            pfc_status = "fail"  # Update global status
    except metadata.PackageNotFoundError:
        # If TTS is not installed
        tts_status = "TTS is not installed."
        pfc_status = "fail"  # Update global status
    tts_markdown = f"""
    ### 🟥 <u>TTS Information</u><br>
    &nbsp;&nbsp;&nbsp;&nbsp; {tts_status_icon} **TTS Version:** {tts_status}
    """
    return tts_markdown

# Disk space check results to append to the Markdown
disk_space_results = check_disk_space()
cuda_results, pytorch_results = generate_cuda_markdown()
system_ram_results = get_system_ram_markdown()
base_model_results = generate_base_model_markdown(base_model_detected)
tts_version_status = check_tts_version()

def pfc_check_fail():
    global pfc_status
    if pfc_status == "fail":
        print("[FINETUNE]")
        print("[FINETUNE] \033[91m****** WARNING PRE-FLIGHT CHECKS FAILED ******* WARNING PRE-FLIGHT CHECKS FAILED *****\033[0m")
        print("[FINETUNE] \033[91m* Please refer to the \033[93mPre-flight check tab \033[91mand resolve any issues before continuing. *\033[0m")
        print("[FINETUNE] \033[91m*********** Expect errors and failures if you do not resolve these issues. ***********\033[0m")
        print("[FINETUNE]")
    return

#####################
#### STEP 1 BITS ####
#####################

def create_temp_folder():
    temp_folder = os.path.join(os.path.dirname(__file__), 'temp_files')
    os.makedirs(temp_folder, exist_ok=True)
    return temp_folder

def create_temporary_file(folder, suffix=".wav"):
    unique_filename = f"custom_tempfile_{int(time.time())}_{random.randint(1, 1000)}{suffix}"
    return os.path.join(folder, unique_filename)



def format_audio_list(target_language, whisper_model, max_sample_length, eval_split_number, speaker_name_input, create_bpe_tokenizer, gradio_progress=gr.Progress()):
    global validate_train_metadata_path, validate_eval_metadata_path, validate_audio_folder, validate_whisper_model, validate_target_language
    buffer = 0.3
    max_duration = float(max_sample_length)  # Ensure max_duration is a float
    eval_percentage = eval_split_number / 100.0
    speaker_name = speaker_name_input
    audio_total_size = 0
    global out_path
    if speaker_name and speaker_name != 'personsname':
        out_path = this_dir / "finetune" / speaker_name
    else:
        out_path = default_path

    os.makedirs(out_path, exist_ok=True)
    temp_folder = os.path.join(out_path, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    audio_folder = os.path.join(out_path, "wavs")
    os.makedirs(audio_folder, exist_ok=True)
    original_samples_folder = os.path.join(out_path, "..", "put-voice-samples-in-here")
    print("[FINETUNE] Preparing Audio/Generating the dataset")

    # Write the target language to lang.txt in the output directory
    lang_file_path = os.path.join(out_path, "lang.txt")
    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()

    if current_language != target_language:
        with open(lang_file_path, 'w', encoding='utf-8') as lang_file:
            lang_file.write(target_language + '\n')
        print("[FINETUNE] Updated lang.txt with the target language.")
    else:
        print("[FINETUNE] The existing language matches the target language")

    gradio_progress((1, 4), desc="Loading in Whisper")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = WhisperModel(whisper_model, device=device, compute_type="float32")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    existing_metadata = {'train': None, 'eval': None}
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")

    if os.path.exists(train_metadata_path):
        existing_metadata['train'] = pd.read_csv(train_metadata_path, sep="|")
        print("[FINETUNE] Existing training metadata found and loaded.")

    if os.path.exists(eval_metadata_path):
        existing_metadata['eval'] = pd.read_csv(eval_metadata_path, sep="|")
        print("[FINETUNE] Existing evaluation metadata found and loaded.")

    # List to store information about files that are too long
    too_long_files = []

    # Process original voice samples to create WAV files
    original_audio_files = [os.path.join(original_samples_folder, file) for file in os.listdir(original_samples_folder) if file.endswith(('.mp3', '.flac', '.wav'))]

    if not original_audio_files:
        print(f"[FINETUNE] No audio files found in {original_samples_folder}. Skipping processing.")
        return None, None, 0

    whisper_words = []
    audio_steps = (0, len(original_audio_files))
    gradio_progress_duration = 0
    gradio_progress(audio_steps, desc="Processing Audio Files", unit="files")

    for audio_path in tqdm(original_audio_files):  # No start argument for tqdm
        start = datetime.datetime.now()
        audio_file_name_without_ext, _ = os.path.splitext(os.path.basename(audio_path))
        temp_audio_path = os.path.join(temp_folder, f"{audio_file_name_without_ext}.wav")

        try:
            shutil.copy2(audio_path, temp_audio_path)
        except Exception as e:
            print(f"[FINETUNE] Error copying file: {e}")
            continue

        wav, sr = torchaudio.load(temp_audio_path, format="wav")
        wav = torch.as_tensor(wav).clone().detach().t().to(torch.float32), sr

        prefix_check = f"wavs/{audio_file_name_without_ext}_"
        skip_processing = False

        for key in ['train', 'eval']:
            if existing_metadata[key] is not None:
                mask = existing_metadata[key]['audio_file'].str.startswith(prefix_check)
                if mask.any():
                    print(f"\n[FINETUNE] Segments from {audio_file_name_without_ext} have been previously processed; skipping...")
                    skip_processing = True
                    audio_total_size = 121
                    break

        if skip_processing:
            continue

        wav, sr = torchaudio.load(audio_path)
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(audio_path, vad_filter=True, word_timestamps=True, language=target_language)
        segments = list(segments)
        i = 1  # Start the index at 1
        sentence = ""
        sentence_start = None
        first_word = True
        words_list = []
        for _, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        for word_idx, word in enumerate(words_list):
            if create_bpe_tokenizer:
                whisper_words.append(word.word)

            if first_word:
                sentence_start = word.start
                if word_idx == 0:
                    sentence_start = max(sentence_start - buffer, 0)
                else:
                    previous_word_end = words_list[word_idx - 1].end
                    sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start) / 2)
                sentence = word.word
                first_word = False
            else:
                sentence += " " + word.word

            # Split segment if it ends with punctuation or exceeds max_duration
            if word.word[-1] in ["!", ".", "?"] or (word.end - sentence_start) > max_duration:
                sentence = sentence.strip()
                # Clean and normalize the sentence
                sentence = multilingual_cleaners(sentence, target_language)
                audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))
                audio_file = f"{audio_file_name}_{str(i).zfill(8)}.wav"

                if word_idx + 1 < len(words_list):
                    next_word_start = words_list[word_idx + 1].start
                else:
                    next_word_start = (wav.shape[0] - 1) / sr

                word_end = min((word.end + next_word_start) / 2, word.end + buffer)

                absolute_path = os.path.join(audio_folder, audio_file)  # Save to the audio_folder
                os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
                first_word = True

                audio = wav[int(sr * sentence_start):int(sr * word_end)].unsqueeze(0)
                
                # If the audio is too long, split it into smaller chunks
                if audio.size(-1) > max_duration * sr:
                    gradio_progress((3,4), "Splitting overly large audio")
                    too_long_files.append((audio_file, audio.size(-1) / sr))
                while audio.size(-1) > max_duration * sr:
                    split_audio = audio[:, :int(max_duration * sr)]
                    audio = audio[:, int(max_duration * sr):]
                    # Normalize the file path
                    split_file_name = f"{audio_file_name}_{str(i).zfill(8)}.wav"
                    split_relative_path = os.path.join(split_file_name)
                    split_absolute_path = os.path.normpath(os.path.join(audio_folder, split_relative_path))
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(split_absolute_path), exist_ok=True)
                    # Save the split audio
                    torchaudio.save(split_absolute_path, split_audio, sr)
                    # Update metadata
                    metadata["audio_file"].append(f"wavs/{split_relative_path}")
                    metadata["text"].append(sentence)
                    metadata["speaker_name"].append(speaker_name)
                    i += 1

                if audio.size(-1) >= sr:  # if the remaining audio is at least half a second long
                    torchaudio.save(absolute_path, audio, sr)
                else:
                    continue

                metadata["audio_file"].append(f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav")
                metadata["text"].append(sentence)
                metadata["speaker_name"].append(speaker_name)
                i += 1  # Increment the index after saving

        os.remove(temp_audio_path)

        end = datetime.datetime.now()
        gradio_progress_duration += (end - start).total_seconds()
        audio_steps = (audio_steps[0] + 1, audio_steps[1])
        additional_data_points_needed = audio_steps[1] - audio_steps[0]
        avg_duration = gradio_progress_duration / audio_steps[0]
        gradio_estimated_duration = (avg_duration * additional_data_points_needed)
        gradio_progress(audio_steps, desc=f"Processing. Estimated Completion: {c_logger.format_duration(gradio_estimated_duration)}", unit="files")

    # Check if the WAV files folder contains files after processing
    audio_files = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith('.wav')]

    if not audio_files:
        print(f"[FINETUNE] No processed audio files found in {audio_folder}. Skipping processing.")
        return None, None, 0

    if os.path.exists(train_metadata_path) and os.path.exists(eval_metadata_path):
        validate_train_metadata_path = train_metadata_path
        validate_eval_metadata_path = eval_metadata_path
        validate_audio_folder = audio_folder
        validate_whisper_model = whisper_model
        validate_target_language = target_language
        
        gradio_progress((4,4), "Finalizing")
        
        del asr_model, existing_metadata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return train_metadata_path, eval_metadata_path, audio_total_size

    # Check if there are any new audio files to process
    if not metadata["audio_file"]:
        print("[FINETUNE] No new audio files to process. Skipping processing.")
        validate_train_metadata_path = train_metadata_path
        validate_eval_metadata_path = eval_metadata_path
        validate_audio_folder = audio_folder
        validate_whisper_model = whisper_model
        validate_target_language = target_language
        
        gradio_progress((4,4), "Finalizing")
        
        del asr_model, existing_metadata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return train_metadata_path, eval_metadata_path, audio_total_size

    new_data_df = pd.DataFrame(metadata)
    combined_train_df_shuffled = new_data_df.sample(frac=1)
    num_val_samples = int(len(combined_train_df_shuffled) * eval_percentage)

    final_eval_set = combined_train_df_shuffled[:num_val_samples]
    final_training_set = combined_train_df_shuffled[num_val_samples:]

    final_training_set.sort_values('audio_file').to_csv(train_metadata_path, sep='|', index=False)
    final_eval_set.sort_values('audio_file').to_csv(eval_metadata_path, sep='|', index=False)

    if create_bpe_tokenizer:
        print(f"Training BPE Tokenizer")
        tokenizer = ByteLevelBPETokenizer(str(this_dir / base_path / "xttsv2_2.0.3" / "vocab.json"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.add_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[STOP]", "[SAPCE]"])
        tokenizer.train_from_iterator(whisper_words, vocab_size=30000, show_progress=True, min_frequency=2, special_tokens=[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[STOP]", "[SPACE]",
        ])
        # Save the tokenizer model
        tokenizer.save(path=str(out_path / "bpe_tokenizer-vocab.json"), pretty=True)

    gradio_progress((4,4), "Finalizing")

    del asr_model, final_eval_set, final_training_set, new_data_df, existing_metadata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print or log the files that were too long
    if too_long_files:
        print(f"[FINETUNE] The following files were too long and were split into smaller chunks:")
        print(f"[FINETUNE]")
        for file_name, length in too_long_files:
            print(f"[FINETUNE] File: {file_name}, Length: {length:.2f} seconds")
        print(f"[FINETUNE]")
    
    validate_train_metadata_path = train_metadata_path
    validate_eval_metadata_path = eval_metadata_path
    validate_audio_folder = audio_folder
    validate_whisper_model = whisper_model
    validate_target_language = target_language
    
    return train_metadata_path, eval_metadata_path, audio_total_size


#######################
# STEP 1 # Validation #
#######################
def normalize_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Convert written numbers to digits
    words = text.split()
    normalized_words = []
    for word in words:
        try:
            # Try to convert word to a number
            normalized_word = str(w2n.word_to_num(word))
        except ValueError:
            # If it fails, keep the original word
            normalized_word = word
        normalized_words.append(normalized_word)
    # Join words back into a single string
    text = ' '.join(normalized_words)
    return text

def get_audio_file_list(mismatches):
    if mismatches.empty:
        return ["No bad transcriptions"]
    else:
        return mismatches["Audio Path"].tolist()

def load_and_display_mismatches():
    def validate_audio_transcriptions(csv_paths, audio_folder, whisper_model, target_language, progress=None):
        metadata_dfs = []
        for csv_path in csv_paths:
            metadata_df = pd.read_csv(csv_path, sep='|')
            metadata_dfs.append(metadata_df) 
        metadata_df = pd.concat(metadata_dfs, ignore_index=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        asr_model = WhisperModel(whisper_model, device=device, compute_type="float32")
        mismatches = []
        missing_files = []
        total_files = metadata_df.shape[0]
        if progress is not None:
            progress((0, total_files), desc="Processing files")
        for index, row in tqdm(metadata_df.iterrows(), total=total_files, unit="file", disable=False, leave=True):
            audio_file = row['audio_file']
            expected_text = row['text']
            audio_file_name = audio_file.replace("wavs/", "")
            audio_path = os.path.normpath(os.path.join(audio_folder, audio_file_name))
            if not os.path.exists(audio_path):
                missing_files.append(audio_file_name)
                if progress is not None:
                    progress((index + 1, total_files), desc="Processing files")  # Update progress bar for skipped files
                continue
            wav, sr = torchaudio.load(audio_path)
            segments, _ = asr_model.transcribe(audio_path, vad_filter=True, word_timestamps=True, language=target_language)
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            normalized_expected_text = normalize_text(expected_text)
            normalized_transcribed_text = normalize_text(transcribed_text)
            if normalized_transcribed_text != normalized_expected_text:
                mismatches.append([
                    row['text'],  # expected_text
                    transcribed_text,
                    audio_file_name,  # Just the filename for display
                    audio_path  # Full path for playback
                ])
            if progress is not None:
                progress((index + 1, total_files), desc="Processing files")  # Update progress bar for each file
        if missing_files:
            print("[FINETUNE]")
            print("[FINETUNE] The following files are missing and should be removed from the CSV files:")
            for file_name in missing_files:
                print(f"[FINETUNE] - {file_name}")                
        return mismatches
    def load_mismatches(csv_paths, audio_folder, whisper_model, target_language, progress):
        mismatches_list = validate_audio_transcriptions(csv_paths, audio_folder, whisper_model, target_language, progress)
        return pd.DataFrame(mismatches_list, columns=["Expected Text", "Transcribed Text", "Filename", "Audio Path"])
    progress = gr.Progress(track_tqdm=True)
    if validate_train_metadata_path and validate_eval_metadata_path and validate_audio_folder and validate_whisper_model and validate_target_language:
        mismatches = load_mismatches([validate_train_metadata_path, validate_eval_metadata_path], validate_audio_folder, validate_whisper_model, validate_target_language, progress)
        file_list = get_audio_file_list(mismatches)
        file_list_select = file_list[0] if file_list else "No Mismatched Audio Files"
        return mismatches[["Expected Text", "Transcribed Text", "Filename"]], gr.Dropdown(choices=file_list, value=file_list_select), ""
    else:
        return pd.DataFrame(columns=["Expected Text", "Transcribed Text", "Filename"]), [], ""


######################
#### STEP 2 BITS #####
######################

from trainer import TrainerArgs, Trainer
from trainer_alltalk.trainer import Trainer

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager


def basemodel_or_finetunedmodel_choice(value):
    global basemodel_or_finetunedmodel 
    if value == "Base Model":
        basemodel_or_finetunedmodel = True
    elif value == "Existing finetuned model":
        basemodel_or_finetunedmodel = False

def train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, learning_rate, model_to_train, continue_run, disable_shared_memory, learning_rate_scheduler, optimizer, num_workers, warm_up, max_audio_length=255995, progress=gr.Progress()):
    pfc_check_fail()

    if "No Models Available" in model_to_train:
        print(f"[FINETUNE] \033[91mError\033[0m: Cannot train. You have not selected a model. Please download a model.")
        print(f"[FINETUNE] \033[91mError\033[0m: into a sub folder within the correct models folder.")
        return   
    #  Logging parameters
    RUN_NAME = "XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None
    model_path = this_dir / "models" / "xtts" / model_to_train

    # Set here the path that the checkpoints will be saved. Default: ./training/
    OUT_PATH = os.path.join(out_path, "training")
    print("[FINETUNE] \033[94mStarting Step 2\033[0m - Fine-tuning the XTTS Encoder")
    print(f"[FINETUNE] \033[94mLanguage: \033[92m{language} \033[94mEpochs: \033[92m{num_epochs} \033[94mBatch size: \033[92m{batch_size}\033[0m \033[94mGrad accumulation steps: \033[92m{grad_acumm}\033[0m")
    print(f"[FINETUNE] \033[94mTraining   : \033[92m{train_csv}\033[0m")
    print(f"[FINETUNE] \033[94mEvaluation : \033[92m{eval_csv}\033[0m")
    print(f"[FINETUNE] \033[94mModel used : \033[92m{model_path}\033[0m")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
		# Get the current device ID
        gpu_device_id = torch.cuda.current_device()
        gpu_available_mem_gb = (torch.cuda.get_device_properties(gpu_device_id).total_memory - torch.cuda.memory_allocated(gpu_device_id)) / (1024 ** 3)
        print(f"[FINETUNE] \033[94mAvailable VRAM: \033[92m{gpu_available_mem_gb:.2f} GB\033[0m")
        if gpu_available_mem_gb < 12:
            print(f"[FINETUNE]")
            print(f"[FINETUNE] \033[91m****** WARNING PRE-FLIGHT CHECKS FAILED ******* WARNING PRE-FLIGHT CHECKS FAILED *****\033[0m")
            print(f"[FINETUNE] \033[94mAvailable VRAM: \033[92m{gpu_available_mem_gb:.2f} GB\033[0m")
            print(f"[FINETUNE] \033[94mIf you are running on a Linux system and you have 12GB's or less of VRAM, this step\033[0m")
            print(f"[FINETUNE] \033[94mmay fail, due to not enough GPU VRAM. Windows systems will use system RAM as extended\033[0m")
            print(f"[FINETUNE] \033[94mVRAM and so should work ok. However, Windows machines will need enough System RAM\033[0m")
            print(f"[FINETUNE] \033[94mavailable. Please read the PFC help section available on the first tab of the web\033[0m")
            print(f"[FINETUNE] \033[94minterface for more information.\033[0m")
            print(f"[FINETUNE] \033[91m****** WARNING PRE-FLIGHT CHECKS FAILED ******* WARNING PRE-FLIGHT CHECKS FAILED *****\033[0m")
            print(f"[FINETUNE]")

    # Create the directory
    os.makedirs(OUT_PATH, exist_ok=True)

    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
    START_WITH_EVAL = False  # if True it will star with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acumm  # set here the grad accumulation steps

    # Define here the dataset that you want to use for the fine-tuning on.
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=train_csv,
        meta_file_val=eval_csv,
        language=language,
    )

    # Add here the configs of the datasets
    DATASETS_CONFIG_LIST = [config_dataset]
    TOKENIZER_FILE = str(model_path / "vocab.json")
    XTTS_CHECKPOINT = str(model_path / "model.pth")
    XTTS_CONFIG_FILE = str(model_path / "config.json")
    DVAE_CHECKPOINT = model_path / "dvae.pth"
    MEL_NORM_FILE = model_path / "mel_stats.pth"
    SPEAKERS_FILE = model_path / "speakers_xtts.pth"

    training_assets = None
    if (out_path / "bpe_tokenizer-vocab.json").exists():
        print("[FINETUNE] Using custom BPE tokenizer")
        training_assets = {
            'Tokenizer': str(out_path / "bpe_tokenizer-vocab.json")
        }

    continue_path = None
    if continue_run:
        folders = glob.glob(os.path.join(OUT_PATH, '*/'))
        if folders:
            last_run = max(folders, key=os.path.getmtime)
            if last_run:
                checkpoints = glob.glob(os.path.join(last_run, "best_model_*.pth"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
                    if latest_checkpoint:
                        XTTS_CHECKPOINT = None
                        continue_path = last_run
                        print(f"[FINETUNE] Continuing previous fine tuning {latest_checkpoint}")


    # Copy the supporting files
    destination_dir = out_path / "chkptandnorm"
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DVAE_CHECKPOINT, destination_dir / DVAE_CHECKPOINT.name)
    shutil.copy2(MEL_NORM_FILE, destination_dir / MEL_NORM_FILE.name)
    shutil.copy2(SPEAKERS_FILE, destination_dir / SPEAKERS_FILE.name)

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=max_audio_length,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    # Resolve Japanese threading issue
    number_of_workers = int(num_workers)
    if language == "ja":
        number_of_workers = 0

    lr_scheduler = None
    lr_scheduler_params = {}

    if learning_rate_scheduler and learning_rate_scheduler != "None":
        lr_gamma_mapping = {
            1e-6: 0.9,
            5e-6: 0.8,
            1e-5: 0.3,
            5e-5: 0.3,
            1e-4: 0.3,
            5e-4: 0.1,
            1e-3: 0.1
        }
        lr_scheduler = learning_rate_scheduler
        if lr_scheduler == "StepLR":
            lr_scheduler_params = {'step_size': 30, 'gamma': 0.1, 'last_epoch': -1}
        elif lr_scheduler == "MultiStepLR":
            exponent = 3 - int(math.log2(num_epochs) / 2)
            base = 2
            num_milestones = min(num_epochs, int(math.pow(base, exponent)))
            milestone_interval = num_epochs // (num_milestones + 1)
            milestones = [milestone_interval * (i + 1) for i in range(num_milestones)]
            lr_scheduler_params = {'milestones': milestones, 'gamma': lr_gamma_mapping[learning_rate], 'last_epoch': -1}
        elif lr_scheduler == "ExponentialLR":
            lr_scheduler_params = {'gamma': 0.5, 'last_epoch': -1}
        elif lr_scheduler == "CosineAnnealingLR":
            lr_scheduler_params = {'T_max': num_epochs, 'eta_min': 1e-6, 'last_epoch': -1}
        elif lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler_params = {'mode': 'min', 'factor': 0.8, 'patience': 1, 'threshold': 0.0001,
                                   'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 1e-8, 'eps': 1e-08,}
        elif lr_scheduler == "CyclicLR":
            lr_scheduler_params = {'base_lr': learning_rate, 'max_lr': 0.1, 'step_size_up': 2000, 'step_size_down': None,
                                   'mode': 'triangular', 'gamma': 1.0, 'scale_fn': None, 'scale_mode': 'cycle',
                                   'cycle_momentum': True, 'base_momentum': 0.8, 'max_momentum': 0.9, 'last_epoch': -1}
        elif lr_scheduler == "OneCycleLR":
            #TODO: Determine total steps / epoch for this
            lr_scheduler_params = {'max_lr': learning_rate, 'total_steps': None, 'epochs_up': None, 'steps_per_epoch': None,
                                   'anneal_strategy': 'cos', 'cycle_momentum': True, 'base_momentum': 0.85,
                                   'max_momentum': 0.95, 'div_factor': 25.0, 'final_div_factor': 10000.0,
                                   'last_epoch': -1}
        elif lr_scheduler == "CosineAnnealingWarmRestarts":
            if num_epochs < 4:
                error_message = "For Cosine Annealing Warm Restarts, epochs must be at least 4. Please set a minimum of 4 epochs."
                progress(1.0, desc=f"Error: {error_message}")
                raise ValueError(error_message)
            #Set 4 learning rate restarts
            lr_scheduler_params = {'T_0': int(num_epochs / 4), 'T_mult': 1, 'eta_min': 1e-6, 'last_epoch': -1}

    optimizer_params = None
    optimizer_wd_only_on_weight = OPTIMIZER_WD_ONLY_ON_WEIGHTS

    if optimizer == "AdamW":
        optimizer_params = {
            "betas": [0.9, 0.96],
            "eps": 1e-8,
            "weight_decay": 1e-2
        }
    elif optimizer == "RMSprop":
        optimizer_params = {
            "alpha": 0.99,
            "eps": 1e-8,
            "weight_decay": 1e-4
        }
    elif optimizer == "SGD":
        optimizer_params = {
            "momentum": 0.9,
            "weight_decay": 1e-4
        }
    elif optimizer == "Adam":
        optimizer_params = {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-4
        }
    elif optimizer == "Adagrad":
        optimizer_params = {
            "lr_decay": 0,
            "weight_decay": 1e-4,
            "eps": 1e-10
        }
    elif optimizer == "RAdam":
        optimizer_params = {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-2
        }
    elif optimizer == "stepwisegraduallr" or optimizer == "noamlr":
        optimizer_params = { }
        # These are Hardcoded learning rate schedulers


    print(f"[FINETUNE] Learning Scheduler {lr_scheduler}, params {lr_scheduler_params}")
    # training parameters config
    config = GPTTrainerConfig(
        epochs=num_epochs,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS training",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=number_of_workers,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer=optimizer,
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params=optimizer_params,
        lr=learning_rate,  # learning rate
        lr_scheduler=lr_scheduler,
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params=lr_scheduler_params,
    progress(0, desc="Model is currently training. See console for more information")
    # init the model from config
    model = GPTTrainer.init_from_config(config)
    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    global c_logger
    c_logger = MetricsLogger()

    # init the trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            continue_path=continue_path,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets=training_assets,
        c_logger=c_logger,
        warmup=warm_up,

    )

    if(disable_shared_memory):
        # Limit training to GPU memory instead of shared memory
        print("[FINETUNE] Limiting GPU memory to 95% to prevent spillover")
        torch.cuda.set_per_process_memory_fraction(0.95)

    trainer.fit()

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx =  samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples, config, model_args, config_dataset
    gc.collect()
    train_samples = None
    eval_samples = None
    config_dataset = None
    trainer = None
    model = None
    model_args = None
    try:
        return XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, trainer_out_path, speaker_ref
    except Exception as e:
        print(f"Error returning values: {e}")
        return "Error", "Error", "Error", "Error", "Error"

##########################
#### STEP 3 AND OTHER ####
##########################

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def find_a_speaker_file(folder_path):
    search_path = folder_path / "*" / "speakers_xtts.pth"
    files = glob.glob(str(search_path), recursive=True)
    latest_file = max(files, key=os.path.getctime, default=None)
    return latest_file

XTTS_MODEL = None
def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    pfc_check_fail()
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "No Models were selected. Click the Refresh Dropdowns button and try again."
    xtts_speakers_pth = find_a_speaker_file(this_dir / "models")
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("[FINETUNE] \033[94mStarting Step 3\033[0m Loading XTTS model!")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False, speaker_file_path=xtts_speakers_pth)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("[FINETUNE] Model Loaded!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None
    speaker_audio_file = str(speaker_audio_file)
    if os.path.isdir(speaker_audio_file):
        wavs_files = glob.glob(os.path.join(speaker_audio_file, "*.wav"))
        speaker_audio_file = wavs_files[0]
    else:
        wavs_files = [speaker_audio_file]

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=wavs_files, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    out = XTTS_MODEL.inference(
        text=tts_text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=XTTS_MODEL.config.temperature, # Add custom parameters here
        length_penalty=XTTS_MODEL.config.length_penalty,
        repetition_penalty=XTTS_MODEL.config.repetition_penalty,
        top_k=XTTS_MODEL.config.top_k,
        top_p=XTTS_MODEL.config.top_p,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_file

def get_available_voices(minimum_size_kb=1200, speaker_name=None):
    directory = out_path
    if speaker_name and speaker_name != 'personsname':
        directory = this_dir / "finetune" / speaker_name

    voice_files = [
        voice for voice in Path(f"{directory}/wavs").glob("*.wav")
        if voice.stat().st_size > minimum_size_kb * 1200  # Convert KB to bytes
    ]
    return sorted([str(file) for file in voice_files])  # Return full path as string

def find_best_models(directory, speaker_name=None):
    if speaker_name and speaker_name != 'personsname':
        directory = this_dir / "finetune" / speaker_name

    """Find files named 'best_model.pth' in the given directory."""
    return [str(file) for file in Path(directory).rglob("best_model.pth")]

def find_models(directory, extension,  speaker_name=None):
    if speaker_name and speaker_name != 'personsname':
        directory = this_dir / "finetune" / speaker_name

    """Find files with a specific extension in the given directory."""
    return [str(file) for file in Path(directory).rglob(f"*.{extension}")]

def find_jsons(directory, filename, speaker_name=None):
    if speaker_name and speaker_name != 'personsname':
        directory = this_dir / "finetune" / speaker_name

    """Find files with a specific filename in the given directory."""
    return [str(file) for file in Path(directory).rglob(filename)]


# XTTS checkpoint files (best_model.pth)
xtts_checkpoint_files = find_best_models(out_path)
# XTTS config files (config.json)
xtts_config_files = find_jsons(out_path, "config.json")
# XTTS vocab files (vocab.json)
xtts_vocab_files = find_jsons(out_path, "vocab.json")

##########################
#### STEP 4 AND OTHER ####
##########################
import shutil
from pathlib import Path
import torch

def compact_custom_model(xtts_checkpoint_copy, folder_path, overwrite_existing):
    this_dir = Path(__file__).parent.resolve()
    if not xtts_checkpoint_copy:
        error_message = "No trained model was selected. Please click Refresh Dropdowns and try again."
        print("[FINETUNE]", error_message)
        return error_message
    target_dir = this_dir / "models" / "xtts" / folder_path
    if overwrite_existing == "Do not overwrite existing files" and target_dir.exists():
        error_message = "The target folder already exists. Please change folder name or allow overwrites."
        print("[FINETUNE]", error_message)
        return error_message
    xtts_checkpoint_copy = Path(xtts_checkpoint_copy)
    try:
        checkpoint = torch.load(xtts_checkpoint_copy, map_location=torch.device("cpu"))
    except Exception as e:
        print("[FINETUNE] Error loading checkpoint:", e)
        raise
    del checkpoint["optimizer"]
    target_dir.mkdir(parents=True, exist_ok=True)
    # Remove dvae-related keys from checkpoint
    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]
    torch.save(checkpoint, target_dir / "model.pth")
    def copy_files(src_folder, dest_folder, files):
        for file_name in files:
            src_path = src_folder / file_name
            dest_path = dest_folder / file_name
            if src_path.exists():
                shutil.copy2(src_path, dest_path)
            else:
                print(f"[FINETUNE] Warning: {src_path} does not exist and will not be copied.")
    # Copy first set of files
    folder_path_new = xtts_checkpoint_copy.parent
    files_to_copy = ["vocab.json", "config.json"]
    copy_files(folder_path_new, target_dir, files_to_copy)
    # Copy second set of files
    chkptandnorm_path = out_path / "chkptandnorm"
    files_to_copy2 = ["speakers_xtts.pth", "mel_stats.pth", "dvae.pth"]
    copy_files(chkptandnorm_path, target_dir, files_to_copy2)
    # Copy large wav files
    source_wavs_dir = out_path / "wavs"
    target_wavs_dir = target_dir / "wavs"
    target_wavs_dir.mkdir(parents=True, exist_ok=True)
    for file_path in source_wavs_dir.iterdir():
        if file_path.is_file() and file_path.stat().st_size > 1000 * 1024:  # File size greater than 1000 KB
            shutil.copy2(file_path, target_wavs_dir / file_path.name)
    print(f"[FINETUNE] Model & Suitable WAV samples copied to '/models/xtts/{folder_path}/'")
    return f"Model & Suitable WAV samples copied to '/models/xtts/{folder_path}/'"

def delete_training_data():
    # Define the folder to be deleted
    folder_to_delete = Path(out_path)

    # Check if the folder exists before deleting
    if folder_to_delete.exists():
        # Iterate over all files and subdirectories
        for item in folder_to_delete.iterdir():
            # Exclude trainer_0_log.txt from deletion
            if item.name != "trainer_0_log.txt":
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except PermissionError:
                    print(f"[FINETUNE] PermissionError: Could not delete {item}. Skipping.")

        print(f"[FINETUNE] Folder {folder_to_delete} contents (excluding trainer_0_log.txt) deleted successfully.")
        return "Folder '/finetune/tmp-trn/' contents (excluding trainer_0_log.txt) deleted successfully."
    else:
        print(f"[FINETUNE] Folder {folder_to_delete} does not exist.")
        return "Folder '/finetune/tmp-trn/' does not exist."

def clear_folder_contents(folder_path):
    # Check if the folder exists before clearing its contents
    if folder_path.exists() and folder_path.is_dir():
        # List all files and subdirectories in the folder
        for item in os.listdir(folder_path):
            item_path = folder_path / item
            if item_path.is_file():
                # If it's a file, remove it
                os.remove(item_path)
            elif item_path.is_dir():
                # If it's a subdirectory, remove it recursively
                shutil.rmtree(item_path)

        print(f"[FINETUNE] Contents of {folder_path} deleted successfully.")
        return f"Contents of '{folder_path}' deleted successfully."
    else:
        print(f"[FINETUNE] Folder {folder_path} does not exist.")
        return f"Folder '{folder_path}' does not exist."

def delete_voice_sample_contents():
    # Define the folders to be cleared
    voice_samples_folder = this_dir / "finetune" / "put-voice-samples-in-here"
    gradio_temp_folder = this_dir / "finetune" / "gradio_temp"
    # Clear the contents of the gradio_temp folder
    gradio_temp_message = clear_folder_contents(gradio_temp_folder)
    # Clear the contents of the voice samples folder
    voice_samples_message = clear_folder_contents(voice_samples_folder)
    return voice_samples_message

#######################
#### OTHER Generic ####
#######################
# define a logger to redirect 
class Logger:
    def __init__(self, filename="finetune.log"):
        self.log_file = filename
        self.terminal = sys.stdout
        self.log = open(self.log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

# redirect stdout and stderr to a file
sys.stdout = Logger()
sys.stderr = sys.stdout

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[FINETUNE] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

c_logger = MetricsLogger()
def load_metrics():
    return c_logger.plot_metrics(), f"Running Time: {c_logger.format_duration(c_logger.total_duration)} - Estimated Completion: {c_logger.format_duration(c_logger.estimated_duration)}"
def read_logs():
    sys.stdout.flush()
    with open(sys.stdout.log_file, "r") as f:
        return f.read()

def cleanup_before_exit(signum, frame):
    print("[FINETUNE] Received interrupt signal. Cleaning up and exiting...")
    # Perform cleanup operations here if necessary
    sys.exit(0)

def create_refresh_button(refresh_components, refresh_methods, elem_class, interactive=True):
    def refresh(speaker_name):
        updates = {}
        for component, method in zip(refresh_components, refresh_methods):
            args = method(speaker_name=speaker_name) if callable(method) else method
            if args and 'choices' in args:
                # Select the most recent file (last in the sorted list)
                args['value'] = args['choices'][-1] if args['choices'] else ""
            for k, v in args.items():
                setattr(component, k, v)
            updates[component] = gr.update(**(args or {}))
        return updates

    refresh_button = gr.Button("Refresh Dropdowns", elem_classes=elem_class, interactive=interactive)
    refresh_button.click(
        fn=refresh,
        inputs=[speaker_name_input_testing],
        outputs=refresh_components
    )

    return refresh_button

def create_refresh_button_next(refresh_components, refresh_methods, elem_class, interactive=True):
    def refresh_export(speaker_name):
        global out_path
        out_path = this_dir / "finetune" / speaker_name
        updates = {}
        for component, method in zip(refresh_components, refresh_methods):
            args = method(speaker_name=speaker_name) if callable(method) else method
            if args and 'choices' in args:
                # Select the most recent file (last in the sorted list)
                args['value'] = args['choices'][-1] if args['choices'] else ""
            for k, v in args.items():
                setattr(component, k, v)
            updates[component] = gr.update(**(args or {}))
        return updates

    refresh_button = gr.Button("Refresh Dropdowns", elem_classes=elem_class, interactive=interactive)
    refresh_button.click(
        fn=refresh_export,
        inputs=[speaker_name_input_export],
        outputs=refresh_components
    )

    return refresh_button


pfc_markdown = f"""
    ### 🚀 <u>Pre-flight Checklist for Fine-tuning</u><br>
    ◽  <strong>Ensure</strong> each criterion is marked with a green check mark ✅ and a Pass status. <strong>Finetuning will fail otherwise.</strong><br>
    ◽  The help tabs along the top will assist in resolving issues and you can also find additional help guides on the AllTalk [GitHub repository](https://github.com/erew123/alltalk_tts#-finetuning-a-model).<br>
    ◽  For an overview of fine-tuning procedures, please refer to the "General Finetuning info" tab or visit the AllTalk [GitHub repository](https://github.com/erew123/alltalk_tts#-finetuning-a-model).
    """

custom_css = """
body {
    font-size: 16px; /* Adjust the base font size as needed */
}
h1, h2, h3, h4, h5, h6 {
    font-size: 1.25em; /* Adjust heading sizes relative to the base size */
}
p {
    font-size: 1.1em; /* Paragraph font size, relative to the base size */
    margin-bottom: 10px; /* Adjust paragraph spacing */
}
.gradio_container {
    zoom: 1.1; /* Adjust the zoom to scale the entire container */
}
"""

if __name__ == "__main__":
    # Register the signal handler
    signal.signal(signal.SIGINT, cleanup_before_exit)

################
#### GRADIO ####
################

    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning demo\n\n"""
        """
        Example runs:
        python3 TTS/demos/xtts_ft_demo/xtts_demo.py --port 
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train. Default: 10",
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size. Default: 4",
        default=4,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Grad accumulation steps. Default: 1",
        default=1,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Max permitted audio size in seconds. Default: 11",
        default=11,
    )

    args = parser.parse_args()

#####################
#### GRADIO INFO ####
#####################

    with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as demo:
        with gr.Row():
            gr.Markdown("## XTTS Models Finetuning")
            gr.Markdown("")
            gr.Markdown("")
            dark_mode_btn = gr.Button("Light/Dark Mode", variant="primary", size="sm")
            dark_mode_btn.click(None, None, None,
            js="""() => {
                if (document.querySelectorAll('.dark').length) {
                    document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
                    // localStorage.setItem('darkMode', 'disabled');
                } else {
                    document.querySelector('body').classList.add('dark');
                    // localStorage.setItem('darkMode', 'enabled');
                }
            }""", show_api=False)
        with gr.Tab("🚀 PFC"):
            with gr.Tab("🚀 Pre-Flight Checklist"):
                gr.Markdown(
                f"""
                {pfc_markdown}       
                {disk_space_results}
                {system_ram_results}
                {cuda_results}
                {pytorch_results}
                {base_model_results}
                {tts_version_status}
                """
            )
            with gr.Tab("🟩 Disks Help"):
                gr.Markdown(
                f"""
                {disk_space_results}<br><br>
                ◽ During actual training (Step 2) Finetuning will require approximately 18GB's of free disk space while performing training and will fail or perform badly if there is any less disk space. The majority of this disk space is used temporarily and will be cleared when you reach Step 4 and move & compact the model then delete the training data.<br>
                ◽ Because lots of data is being copied around, <strong>mechanical hard disks</strong> will be slow.
                """
            )
            with gr.Tab("🟪 RAM & VRAM Help"):
                gr.Markdown(
                f"""
                {system_ram_results}<br>
                ◽ During actual training (Step 2) Finetuning will use around **14GB's** of VRAM. If your GPU doesnt have 14GB's of VRAM:<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **Windows** systems will attempt to extend VRAM into your System RAM, and so should work.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **Linux** systems can only use the available VRAM, so may fail on 12GB VRAM or smaller GPU's.<br>
                ◽ For **Windows** users with 12GB or less, if you also have very low or slow System RAM you can expect bad performance or the training to fail.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - 12GB cards may need 2GB System RAM.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - 8GB cards may need 6GB System RAM.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - 6GB cards may need 8GB System RAM.<br>
                ◽ Its hard to estimate what performance impact this could will have, due to different memory speeds, PCI speeds, GPU speeds etc.<br>
                ◽ If you have a low VRAM scenario and also are attempting to run on a mechanical hard drive, Finetuning could take ??? amount of time.<br>
                ◽ On Windows machines, please ensure you have **not** disabled System Memory Fallback for Stable Diffusion <a href="https://nvidia.custhelp.com/app/answers/detail/a_id/5490/~/system-memory-fallback-for-stable-diffusion" target="_blank">link here</a><br>
                """
            )
            with gr.Tab("🟨 CUDA & Cublas Help"):
                gr.Markdown(
                f"""         
                {cuda_results}<br><br>
                ◽ It DOESNT matter what version of CUDA you have installed within Python either, CUDA 11.8, CUDA 12.1 etc. The NVIDIA CUDA Development Toolkit is a completly different and seperate thing from Python/PyTorch.<br>
                ◽ Finetuning simply wants to access a tool within the CUDA Development Toolkit 11.8 called Cublas64_11.<br>
                ◽ If you dont have the toolkit installed, the idea is just to install the smallest bit possible and this will not affect or impact other things on your system.<br><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ You will need to download the Nvidia Cuda Toolkit 11.8<span style="color: #3366ff;"> network install</span> from <a href="https://developer.nvidia.com/cuda-11-8-0-download-archive" target="_blank">link here</a><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ 1) Run the installer and select <span style="color: #3366ff;">Custom Advanced</span> Uncheck <span style="color: #3366ff;">everything</span> at the top then expand <span style="color: #3366ff;">CUDA</span>, <span style="color: #3366ff;">Development</span> > <span style="color: #3366ff;">Compiler</span> > and select <span style="color: #3366ff;;">nvcc</span> then expand <span style="color: #3366ff;;">Libraries</span> and select <span style="color: #3366ff;;">CUBLAS</span>.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ 2) Back at the top of <span style="color: #3366ff;">CUDA</span>, expand <span style="color: #3366ff;">Runtime</span> > <span style="color: #3366ff;">Libraries</span> and select <span style="color: #3366ff;">CUBLAS</span>. Click <span style="color: #3366ff;;">Next</span>, accept the default path (taking a note of its location) and let the install run. <br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ 3) You should be able to drop to your terminal or command prompt and type <span style="color: #3366ff;">nvcc --version</span> and have it report <span style="color: #00a000;">Cuda compilation tools, release 11.8</span>. If it does you are good to go. If it doesn't > Step 4.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ 4) <strong>Linux users</strong>, you can temporarily add these paths on your current terminal window with (you may need to confirm these are correct for your flavour of Linux):<br><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #3366ff;">export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64&colon;&dollar;&lbrace;LD_LIBRARY_PATH&colon;&plus;&colon;&dollar;&lbrace;LD_LIBRARY_PATH&rbrace;&rbrace;</span> (Add it to your ~/.bashrc if you want this to be permanent)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #3366ff;">export LD_LIBRARY_PATH=/usr/local/cuda-11.8/bin</span><br><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>Windows users</strong> need the add the following to the PATH environment variable. Start menu and search for "Environment Variables" or "Edit the system environment variables.". <br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Find and select the "Path" variable, then click on the "Edit...". Click on the "New" button and add:<br><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #3366ff;">C:&bsol;Program Files&bsol;NVIDIA GPU Computing Toolkit&bsol;CUDA&bsol;v11.8&bsol;bin.</span><br><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ 5) Once you have these set correctly, you should be able to open a new command prompt/terminal and <span style="color: #3366ff;">nvcc --version</span> at the command prompt/terminal, resulting in <span style="color: #00a000;">Cuda compilation tools, release 11.8</span>.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ 6) If the nvcc command doesn't work OR it reports a version different from 11.8, finetuning wont work, so you will to double check your environment variables and get them working correctly.<br>
                """
            )
            with gr.Tab("🟦 Python & PyTorch Help"):
                gr.Markdown(
                f"""         
                {pytorch_results}<br><br>
                ◽ On the PyTorch version the:<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- first few digits are the version of PyTorch e.g. 2.1.0 is PyTorch 2.1.0<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- last few digits refer to the CUDA version e.g. cu118 is Cuda 11.8. cu121 is Cuda 12.1.<br>
                ◽ Ensure you have started your Python envuronment before running finetuning otherwise you will have failures on the above checks.<br>
                ◽ If PyTorch does not show a CUDA version, then PyTorch will need reinstalling with CUDA. I would suggest running <span style="color: #3366ff;">pip cache purge</span> before installing PyTorch again.<br>
                ◽ It DOESNT matter what version of PyTorch and CUDA you have installed within Python, CUDA 11.8, CUDA 12.1 etc. The NVIDIA CUDA Development Toolkit is a completly different and seperate thing.<br>
                ◽ Finetuning simply wants to access a tool within the CUDA Development Toolkit called Cublas64_11.<br>
                ◽ If you dont have the toolkit installed, the idea is just to install the smallest bit possible and this will not affect or impact other things on your system.<br>
                """
            )
            with gr.Tab("⬛ XTTS Base Model Help"):
                gr.Markdown(
                f"""         
                {base_model_results}<br><br>
                ◽ If your basemodel is not being detected, please ensure that <span style="color: #3366ff;">finetune.py</span> is being run from the AllTalk main folder.<br>
                ◽ Ensure you have started AllTalk normally at least once and downlaoded a XTTS model.<br>
                ◽ Check that there is an XTTS model within the models folder e.g. <span style="color: #3366ff;">/models/xtts/model-folder-here/</span><br>
                ◽ The files required inside the folder are "model.pth", "vocab.json", "config.json", "dvae.pth", "mel_stats.pth", "speakers_xtts.pth".
                """
            )
            with gr.Tab("🟥 TTS Version Help"):
                gr.Markdown(
                f"""         
                {tts_version_status}<br><br>
                ◽ If your TTS version is showing as the incorrect version, please reinstall the Finetuning requirements at the command prompt/terminal.<br>
                ◽ <span style="color: #3366ff;">pip install -r requirements_finetune.txt</span><br>
                """
            )
                
        with gr.Tab("ℹ️ General Finetuning info"):
            gr.Markdown(
            f"""
            ### 🟥 <u>Important Note</u>
            ◽ <span style="color: #3366ff;">finetune.py</span> needs to be run from the <span style="color: #3366ff;">/alltalk_tts/</span> folder. Don't move the location of this script.
            ### 🟦 <u>What you need to run finetuning</u>
            ◽ An Nvidia GPU.<br>
            ◽ If you have multiple Nvidia GPU's in your system, please see the Github Help section [Multiple GPU's](https://github.com/erew123/alltalk_tts#performance-and-compatibility-issues).<br>
            ◽ Some decent quality audio, multiple files if you like. Minimum of 2 minutes and Ive tested up to 20 minutes of audio.<br>
            ◽ There is no major need to chop down your audio files into small slices as Step 1 will do that for you automatically and prepare the training set. But this can be helpful in cases where:<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Whisper doesnt correctly detect and split down audio files.<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - You do not get a lot of "Speaker Reference Audio" files at the end of training.<br>
            ◽ This process will need access to all your GPU and VRAM, so close any other software that's using your GPU currently.<br>
            ### 🟨 <u>What do I do from here?</u><br>
            ◽ Proceed through Step 1, 2, 3 and onto "What to do next".<br>
            ### 🟩 <u>Additional Information</u><br>
            ◽ Guidance is provided on each step of the process however, if you are after more detailed information please visit:<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◽ [AllTalk Github Finetuning](https://github.com/erew123/alltalk_tts#-finetuning-a-model)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◽ [Coqui XTTS Documentation](https://docs.coqui.ai/en/latest/index.html)<br>
            """
        )

#######################
#### GRADIO STEP 1 ####
#######################
        with gr.Tab("📁 Step 1 - Generating the dataset"):
            with gr.Tab("Generate Dataset"):  
                # out_path = gr.Textbox(
                #     label="Output path (where data and checkpoints will be saved):",
                #     value=out_path,
                #     visible=False,
                # )
                # Define directories
                this_dir = Path(__file__).parent.resolve()
                voice_samples_dir = this_dir / "finetune" / "put-voice-samples-in-here"
                training_data_dir = this_dir / "finetune" / "tmp-trn" / "wavs"
                metadata_files = [
                    this_dir / "finetune" / "tmp-trn" / "metadata_eval.csv",
                    this_dir / "finetune" / "tmp-trn" / "metadata_train.csv",
                    this_dir / "finetune" / "tmp-trn" / "lang.txt"
                ]

                # Ensure directories exist
                voice_samples_dir.mkdir(parents=True, exist_ok=True)
                training_data_dir.mkdir(parents=True, exist_ok=True)

                def upload_audio(files):
                    for file in files:
                        shutil.copy(file.name, voice_samples_dir)
                    return f"Uploaded {len(files)} files to {voice_samples_dir}"

                def delete_existing_audio():
                    for file in voice_samples_dir.iterdir():
                        if file.is_file():
                            file.unlink()
                    return f"Deleted all files in {voice_samples_dir}"

                def delete_existing_training_data():
                    if training_data_dir.exists():
                        for file in training_data_dir.iterdir():
                            if file.is_file():
                                file.unlink()
                    for metadata_file in metadata_files:
                        if metadata_file.exists():
                            metadata_file.unlink()
                    return f"Deleted all files in {training_data_dir} and related metadata files"

                with gr.Row():
                    with gr.Column(scale=1):
                        audio_files_upload = gr.Files(label="Upload Audio Files")
                    with gr.Column(scale=3):
                        with gr.Row():                                        
                            with gr.Column(scale=1):                        
                                audio_upload_button = gr.Button("Upload New Audio Samples")
                                delete_audio_button = gr.Button("Delete Existing Audio Samples")
                                delete_dataset_button = gr.Button("Delete Existing Training Dataset")                        
                            with gr.Column(scale=2):
                                gr.Markdown("""
                                You can manually copy your audio files to `/finetune/put-voice-samples-in-here/` or use the upload to the left and click "Upload New Audio Samples". Once you have uploaded files, you can start creating your dataset.

                                - If you wish to delete previously uploaded audio samples files then use 'Delete Existing Audio Samples'.
                                - If you wish to delete previously generated training datasets, please use 'Delete Existing Training Dataset'.
                                - If you wish to re-use your previously created training data, just click 'Create Dataset'.
                                """)
                        with gr.Row():
                            output_text = gr.Textbox(label="Audio File Management Result", interactive=False) 

                # Define actions for buttons
                audio_upload_button.click(upload_audio, inputs=audio_files_upload, outputs=output_text)
                delete_audio_button.click(delete_existing_audio, outputs=output_text)
                delete_dataset_button.click(delete_existing_training_data, outputs=output_text)
                
                with gr.Row():
                    speaker_name_input = gr.Textbox(
                        label="Training Project Name",
                        value="personsname",
                        visible=True,
                        scale=2,
                    )
                    whisper_model = gr.Dropdown(
                        label="Whisper Model",
                        value="distil-large-v3",
                        choices=[
                            ("large-v3", "large-v3"),
                            ("large-v2", "large-v2"),
                            ("large", "large"),
                            ("medium", "large"),
                            ("small", "large"),
                            ("distil-large-v3 (en only)", "distil-large-v3"),
                            ("distil-large-v2 (en only)", "distil-large-v2"),
                        ],
                        scale=2,
                    )

                    lang = gr.Dropdown(
                        label="Dataset Language",
                        value="en",
                        choices=[
                            "en",
                            "es",
                            "fr",
                            "de",
                            "it",
                            "pt",
                            "pl",
                            "tr",
                            "ru",
                            "nl",
                            "cs",
                            "ar",
                            "zh",
                            "hu",
                            "ko",
                            "ja"
                        ],
                        scale=1,
                    )
                    max_sample_length = gr.Dropdown(
                        label="Max Audio Length (in seconds)",
                        value="30",
                        choices=["10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90"],
                        scale=2
                    )
                    eval_split_number = gr.Number(
                        label="Evaluation Data Split",
                        value=15,  # Default value
                        minimum=5,  # Minimum value
                        maximum=95,  # Maximum value
                        step=1,  # Increment step
                        scale=1,
                    )
                    create_bpe_tokenizer = gr.Checkbox(
                        label='BPE Tokenizer',
                        value=False,
                        info='Custom Tokenizer for training'
                    )
                progress_data = gr.Label(
                    label="Progress:"
                )
                logs = gr.Textbox(
                    label="Logs:",
                    interactive=False,
                    lines=10,
                )
                demo.load(read_logs, None, logs, every=1)

                prompt_compute_btn = gr.Button(value="Step 1 - Create dataset")
            
                def preprocess_dataset(language, whisper_model, max_sample_length, eval_split_number, speaker_name_input, create_bpe_tokenizer, progress=gr.Progress()):
                    clear_gpu_cache()
                    test_for_audio_files = [file for file in os.listdir(audio_folder) if any(file.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac'])]
                    if not test_for_audio_files:
                        return "I cannot find any mp3, wav or flac files in the folder called 'put-voice-samples-in-here'", "", ""
                    else:
                        try:

                            train_meta, eval_meta, audio_total_size = format_audio_list(target_language=language, whisper_model=whisper_model, max_sample_length=max_sample_length, eval_split_number=eval_split_number, speaker_name_input=speaker_name_input, create_bpe_tokenizer=create_bpe_tokenizer, gradio_progress=progress)
                        except:
                            traceback.print_exc()
                            error = traceback.format_exc()
                            return f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}", "", ""

                    clear_gpu_cache()

                    # if audio total len is less than 2 minutes raise an error
                    if audio_total_size < 120:
                        message = "The total duration of the audio file or files you provided, was less than 2 minutes in length. Please add more audio samples."
                        print("[FINETUNE] ",message)
                        return message, "", ""

                    print("[FINETUNE] Dataset Generated. Either run Dataset Validation or move to Step 2")
                    return "Dataset Generated. Either run Dataset Validation or move to Step 2", train_meta, eval_meta, speaker_name_input
            
            def play_selected_audio(selected_file):
                if selected_file and selected_file != "Please Generate Your Dataset":
                    return selected_file
                return None
        
            with gr.TabItem("Dataset Validation"):
                gr.Markdown("""# Audio Transcription Validation
                            This feature allows you to validate the transcriptions in your dataset by comparing the generated audio files against the expected transcriptions provided in the training and evaluation CSV files. It uses the Whisper model to transcribe each audio file and checks if the transcribed text matches the corresponding text in the CSV files. Any mismatches found during the validation process are highlighted in a list, and you can select and play the mismatched audio files from a dropdown menu to manually compare them with the transcriptions. If necessary, you'll need to manually edit and update the CSV files to correct any discrepancies identified during the validation process. Whisper is a Best Effort helper for generating a dataset and editing the dataset or manual dataset generation will be the only way to get a 100% perfect dataset.
                            """)
                
                with gr.Row():
                    load_button = gr.Button("Run Validation")
                    progress_box = gr.Textbox(label="Progress", interactive=False)
                with gr.Row():
                    audio_dropdown = gr.Dropdown(label="Select Audio File to Play", choices=["Please Generate Your Dataset"], value="Please Generate Your Dataset", allow_custom_value=True)
                    audio_output = gr.Audio(label="Audio Player", interactive=False)
                
                mismatch_table = gr.Dataframe(headers=["Text in the CSV Files", "Whisper Transcribed Text", "Filename"], datatype=["str", "str", "str"], interactive=False, wrap=True)
            
                load_button.click(load_and_display_mismatches, [], [mismatch_table, audio_dropdown, progress_box])           
                audio_dropdown.change(play_selected_audio, [audio_dropdown], audio_output)
            
            with gr.Tab("Generate Dataset Instructions"):
                gr.Markdown(
                    f"""
                    #### 🟦 <u>What You Need to Do</u>
                    1. Read [Coqui's guide on creating a good dataset](https://docs.coqui.ai/en/latest/what_makes_a_good_dataset.html).<br>
                    2. Place your audio files (MP3, WAV, or FLAC) in `{str(audio_folder)}` Or Upload them through the interface.<br>
                    3. Provide at least 2 minutes of audio, preferably 5-10 minutes for better results.<br>
                    4. Maximum Audio Length will ensure the audio samples do not exceed the set value in seconds. The AI model can only process so much audio per audio sample, per epoch.<br>
                    5. Enter a unique Project Name if you wish to create a dedicated training folder. You will need to specify this name as you move through the interfaces and click the Refresh buttons where needed.<br>
                    6. Select the Whisper model (distil-large-v3 recommended for English use cases).<br>
                    7. Choose your Dataset Language.<br>
                    8. Adjust the Evaluation Split percentage (default 15% is suitable for most cases).<br>
                    9. Optionally enable the BPE Tokenizer for custom tokenization, which can improve the model's ability to handle unique words, accents, or languages by creating a vocabulary tailored to your specific dataset. This is especially useful for non-standard speech patterns or languages with complex morphology.<br>

                    #### 🟨 <u>What This Step Does</u>
                    1. Uses Whisper to transcribe and segment your audio files.<br>
                    2. Creates smaller audio clips and corresponding transcriptions.<br>
                    3. Generates two CSV files: `metadata_train.csv` and `metadata_eval.csv`.<br>
                    4. Saves processed audio in `/finetune/[project_name]/wavs/`.<br>
                    5. Stores CSV files and metadata in `/finetune/[project_name]/`.<br>

                    Key files generated:<br>
                    ◽ `/finetune/[project_name]/lang.txt`: Contains the two-letter language code.<br>
                    ◽ `/finetune/[project_name]/metadata_train.csv`: List of audio files and transcriptions for training.<br>
                    ◽ `/finetune/[project_name]/metadata_eval.csv`: Evaluation data for assessing training quality.<br>

                    #### <u>Important Considerations</u>
                    ◽ For multi-speaker audio, manually edit CSV files to remove other speakers.<br>
                    ◽ You can manually edit CSV files to improve transcription accuracy & use the `Dataset Validation` to help identify possible bad transcriptions.<br>
                    ◽ The Whisper 2 model may provide better results in audio splitting and dataset creation for English languages, however please see the Whisper site for more details on other languages.<br>
                    ◽ Each Whisper model is about 3GB. It's recommended to stick with one model to save disk space.<br>
                    """
                )
                
            with gr.Tab("Generate Dataset Notes"):
                gr.Markdown(
                    f"""
                    #### 🟥 <u>Important Notes</u>
                    ◽ **Windows Users:** If you encounter a `UserWarning: huggingface_hub cache-system uses symlinks` error, restart your Windows command prompt with `Run as Administrator` and relaunch the finetuning process. This typically only occurs during the first Whisper model download.<br>
                    ◽ **Language Support:** Whisper's performance varies across languages. The Large-v3 model may offer better results for certain languages. For problematic languages, consider manual dataset creation.<br>

                    #### 🟩 <u>Processing Time</u>
                    ◽ First run: Requires downloading the 3GB Whisper model.<br>
                    ◽ Subsequent runs: A few minutes on an average 3-4 year old system.<br>

                    After processing, use the 'Dataset Validation' tab to check transcription accuracy and make necessary corrections.<br>

                    For more details on Whisper models or manual dataset creation, refer to:<br>
                    ◽ [Whisper Models Information](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages)<br>
                    ◽ [Manual Dataset Creation Guide](https://docs.coqui.ai/en/latest/formatting_your_dataset.html)
                    """
                )                
                
                
            
#######################
#### GRADIO STEP 2 ####
#######################
        with gr.Tab("💻 Step 2 - Training"):
            with gr.Tab("Training the model"):
                with gr.Row():
                    speaker_name_input_training = gr.Textbox(
                        label="Project Name",
                        value="personsname",
                        visible=True,
                        scale=1,
                    )
                    with gr.Group():
                        continue_run = gr.Checkbox(
                            value=False,
                            label="Continue Previous Project",
                            scale=1,
                        )
                        disable_shared_memory = gr.Checkbox(
                            value=False,
                            label="Disable Shared Memory Use",
                            scale=1,
                        )
                        warm_up = gr.Checkbox(
                            value=False,
                            label="Perform Small a Warmup Learning",
                            scale=1,
                        )
                    train_csv = gr.Textbox(
                        label="Train CSV:",
                        scale=2,
                    )
                    eval_csv = gr.Textbox(
                        label="Eval CSV:",
                        scale=2,
                    )
                with gr.Row():
                    model_to_train_choices = list(available_models.keys())
                    model_to_train = gr.Dropdown(
                        choices=model_to_train_choices,
                        label="Select the Model to train",
                        value=model_to_train_choices[0] if model_to_train_choices else None,
                        scale=2,
                    )                  
  
                    learning_rates = gr.Dropdown(
                        value=5e-6,
                        label="Learning Rate",
                        choices=[
                            ("1e-6", 1e-6),
                            ("5e-6", 5e-6),
                            ("1e-5", 1e-5),
                            ("5e-5", 5e-5),
                            ("1e-4", 1e-4),
                            ("5e-4", 5e-4),
                            ("1e-3", 1e-3),
                        ],
                        type="value",
                        allow_custom_value=True,
                        scale=1,
                    )
                    learning_rate_scheduler = gr.Dropdown(
                        value="CosineAnnealingWarmRestarts",
                        label="Learning Rate Scheduler(s)",
                        choices=[
                            ("None", "None"),
                            ("Cosine Annealing", "CosineAnnealingLR"),
                            ("Cosine Annealing Warm Restarts", "CosineAnnealingWarmRestarts"),
                            ("Cyclic", "CyclicLR"),
                            ("Exponential", "ExponentialLR"),
                            ("Multi Step", "MultiStepLR"),
                            ("Reduce on Plateau", "ReduceLROnPlateau"),
                            ("Step", "StepLR")
                            #("OneCycleLR", "OneCycleLR"),
                        ],
                        type="value",
                        allow_custom_value=False,
                        scale=2,
                    )
                    optimizer = gr.Dropdown(
                        value="AdamW",
                        label="Optimizer",
                        choices=[
                            ("AdamW", "AdamW"),
                            ("SGD with Momentum", "SGD"),
                            ("RMSprop", "RMSprop"),
                            ("Adagrad", "Adagrad"),
                            ("Adam", "Adam"),
                            ("Rectified Adam", "RAdam"),
                            ("Step Wise", "stepwisegraduallr"),
                            ("Noam", "noamlr")
                        ],
                        type="value",
                        allow_custom_value=False,
                        scale=2,
                    )
                    num_workers = gr.Dropdown(
                        value="8",
                        label="Workers/Threads",
                        choices=[
                            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
                        ],
                        allow_custom_value=False,
                        interactive=True,                     
                        scale=0,
                    )
                with gr.Row():
                    num_epochs =  gr.Slider(
                        label="Number of epochs:",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=args.num_epochs,
                    )
                    batch_size = gr.Slider(
                        label="Batch size:",
                        minimum=2,
                        maximum=512,
                        step=1,
                        value=args.batch_size,
                    )
                    grad_acumm = gr.Slider(
                        label="Grad accumulation steps:",
                        minimum=1,
                        maximum=128,
                        step=1,
                        value=args.grad_acumm,
                    )
                    max_audio_length = gr.Slider(
                        label="Max permitted audio size in seconds:",
                        minimum=2,
                        maximum=20,
                        step=1,
                        value=args.max_audio_length,
                    )

                progress_train = gr.Label(
                    label="Progress:"
                )

                with gr.Row():
                    train_time = gr.Label("Estimated Total Training Time", show_label=False, scale=2)
                    train_btn = gr.Button(value="Step 2 - Run the training", scale=1)

                with gr.Row():
                    model_data = gr.Image(c_logger.plot_metrics(), show_label=False)

                logs_tts_train = gr.Textbox(
                    label="Logs:",
                    interactive=False,
                    lines=10,
                )
                demo.load(load_metrics, None, [model_data, train_time], every=1)
                demo.load(read_logs, None, logs_tts_train, every=1)

                def train_model(language, train_csv, eval_csv, learning_rates, model_to_train, num_epochs, batch_size, grad_acumm, max_audio_length, speaker_name_input_training, continue_run, disable_shared_memory, learning_rate_scheduler, optimizer, num_workers, warm_up, progress=gr.Progress()):
                    clear_gpu_cache()
                    global out_path
                    if speaker_name_input_training and speaker_name_input_training != 'personsname':
                        out_path = this_dir / "finetune" / speaker_name_input_training
                    else:
                        out_path = default_path

                    if not train_csv or not eval_csv:
                        if (out_path / "metadata_eval.csv").exists() and (out_path / "metadata_train.csv").exists():
                            train_csv = out_path / "metadata_train.csv"
                            eval_csv = out_path / "metadata_eval.csv"
                            print("Using existing metadata and training csv.")
                        else:
                            return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
                    try:
                        # convert seconds to waveform frames
                        max_audio_length = int(max_audio_length * 22050)
                        learning_rate = float(learning_rates)  # Convert the learning rate value to a float
                        progress(0, "Initializing training...")
                        config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, learning_rate, model_to_train, continue_run, disable_shared_memory, learning_rate_scheduler, optimizer, num_workers, warm_up, max_audio_length=max_audio_length, progress=gr.Progress())
                    
                        # copy original files to avoid parameters changes issues
                        shutil.copy(config_path, exp_path)
                        shutil.copy(vocab_file, exp_path)

                        ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
                        print("[FINETUNE] Model training done. Move to Step 3")
                        clear_gpu_cache()
                        return "Model training done. Move to Step 3", config_path, vocab_file, ft_xtts_checkpoint, speaker_wav, speaker_name_input_training

                    except ValueError as ve:
                        # This will catch the specific ValueError we raised in train_gpt
                        error_message = str(ve)
                        print(f"[FINETUNE] Error: {error_message}")
                        return f"Training error: {error_message}", "", "", "", "", ""
                    except Exception as e:
                        # This will catch any other unexpected errors
                        error_message = f"An unexpected error occurred: {str(e)}"
                        print(f"[FINETUNE] Error: {error_message}")
                        return f"Training error: {error_message}", "", "", "", "", ""
                
            with gr.Tab("Training Instructions"):
                gr.Markdown(
                    f"""
                    ### 🟥 <u>Important Note - Language support.</u>
                    ◽ If this step is failing/erroring you may wish to check your training data was created correctly (Detailed in Step 1), confirming that wav files have been generated and your `metadata_train.csv` and `metadata_eval.csv` files have been populated.<br>             
                    ◽ Ignore 'fatal: not a git repository (or any of the parent directories): .git' as this is a legacy training script issue.<br>  
                    ### 🟦 <u>What you need to do</u>
                    ◽ The <span style="color: #3366ff;">Train CSV</span> and <span style="color: #3366ff;">Eval CSV</span> should already be populated. If not, just go back to Step 1 and click "Create Dataset" again.<br>
                    ◽ The default settings below are the suggested settings for most purposes, however you may choose to alter them depending on your specific use case.<br>
                    ◽ There are no absolute correct settings for training. It will vary based on:<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- What you are training (A human voice, A cartoon voice, A new language entirely etc)<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- How much audio you have (you may want less or more eval or epochs)<br>
                    ◽ The key indicator of sufficient model training is whether it sounds right to you. Coqui suggests beginning with the base settings for training. If the resulting audio doesn't meet your expectations, additional training sessions may be necessary.<br>
                    #### 🟪 <u>Continue Previous Project</u>
                    This option allows you to resume training from where it left off if your previous session was interrupted (e.g., due to a power outage or system crash). When enabled:<br>
                    ◽ The system will attempt to load the most recent checkpoint from your previous training session.<br>
                    ◽ It will continue training from that point, preserving your progress and saving time.<br>
                    ◽ This is particularly useful for long training sessions or when you want to extend training on a previously trained model.<br>
                    ### 🟨 <u>What this step is doing</u><br>
                    ◽ Very simply put, it's taking all our wav files generated in Step 1, along with our recorded speech in out excel documents and its training the model on that voice e.g. listen to this audio file and this is what is being said in it, so learn to reproduce it.<br>
                    ### 🟩 <u>How long will this take?</u><br>
                    ◽ On a RTX 4070 with factory settings (as below) and 20 minutes of audio, this took 20 minutes. Again, it will vary by system and how much audio you are throwing at it. You have time to go make a coffee.<br>
                    ◽ Look at your command prompt/terminal window if you want to see what it is doing.<br>
                    """
                )                    
            
            with gr.Tab("Memory Optimization"):
                gr.Markdown(
                    f"""
                    ### 🟦 <u>Memory Optimization Strategies</u>

                    Adjusting these parameters can help manage memory usage during training:

                    ◽ **Batch Size**: Smaller batch sizes use less memory. If you encounter out-of-memory errors, try reducing the batch size.<br>
                    ◽ **Gradient Accumulation**: Increasing grad accumulation steps allows you to simulate larger batch sizes without increasing memory usage.<br>
                    ◽ **Workers/Threads**: Fewer workers generally use less memory. If you're experiencing memory issues, try reducing the number of workers.<br>
                    ◽ **Max Audio Length**: Limiting the maximum audio length reduces memory usage by preventing very long audio clips from being loaded.<br>
                    ◽ **Learning Rate and Optimizer**: While these don't directly affect memory usage, they can impact training efficiency. A well-tuned learning rate and appropriate optimizer can lead to faster convergence, potentially reducing overall training time and resource usage.<br>
                    
                    #### 🟪 <u>Disable Shared Memory</u>
                    This option affects how the system manages memory during training:<br>
                    ◽ When enabled, it limits GPU memory usage to 95% of its capacity.<br>
                    ◽ This prevents the training process from spilling over into your system's shared memory.<br>
                    ◽ You're more likely to get a clear "out of memory" error if your GPU runs out of memory, rather than experiencing slower performance due to memory spillover.<br>
                    ◽ This option is useful if you prefer to keep training confined to GPU memory for better performance and clearer error reporting.<br>

                    Important Note for Windows Users with Low VRAM GPUs:<br>
                    ◽ If you're using Windows and have a GPU with lower VRAM (e.g., 8GB or less), it's recommended to leave this option disabled.<br>
                    ◽ Windows can utilize system RAM as extended VRAM, which can be crucial for completing the training process on lower VRAM GPUs.<br>
                    ◽ Disabling this option allows Windows to manage memory allocation more flexibly, potentially preventing out-of-memory errors on GPUs with limited VRAM.
                    

                    Remember, the goal is to find a balance between memory usage and training effectiveness. Start with lower values and gradually increase them while monitoring your system's memory usage.
                    """
                )
      
            with gr.Tab("Info - Grad Accumulation"):
                                gr.Markdown(
                    f"""
                    ### 🟩 <u>Grad Accumulation</u><br>
                    ◽ Gradient accumulation is a technique that allows you to simulate a larger batch size without actually increasing the memory consumption. Here's how it works:<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ Instead of updating the model's parameters after every batch, gradient accumulation enables you to process multiple batches and accumulate their gradients.<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ By setting steps greater than 1, you can process multiple batches before a single optimization step e.g if steps is set to 4, the gradients will be accumulated over 4 batches before updating the model's parameters.<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ This means that you can effectively use a larger batch size while keeping the actual batch size per iteration smaller, thereby reducing the memory footprint.<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ Increasing the steps allows you to find a balance between memory consumption and computational efficiency. You can process more examples per optimization step without exceeding the available memory.<br><br>
                    ◽ However, it's important to note that increasing the gradient accumulation steps does have an impact on the training process:<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ Since the model's parameters are updated less frequently (every grad_accum_steps batches), the training dynamics may be slightly different compared to updating the model after every batch.<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ You may need to adjust the learning rate and other hyperparameters accordingly to compensate for the less frequent updates. Typically, you can increase the learning rate slightly when using gradient accumulation.<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ The training progress will be slower in terms of the number of optimization steps per epoch, as the model updates occur less frequently.<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ However, the overall training time may still be reduced compared to using a smaller batch size without gradient accumulation, as it allows for better utilization of GPU resources.<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◽ To start, you can try setting steps to a value like 4 and see if it resolves the OOM error. If the error persists, you can experiment with higher values until you find a balance that works for your specific setup.
                    """
                )
            with gr.Tab("Info - Batch Size"):
                                gr.Markdown(
                    f"""
                    ### 🟩 <u>Batch Size</u><br>
                    ◽ Batch size refers to the number of training examples processed in a single iteration (forward and backward pass) before updating the model's parameters.<br>
                    ◽ A larger batch size allows for more efficient computation and utilization of hardware resources, especially when using GPUs.<br>
                    ◽ However, increasing the batch size also increases the memory consumption during training, as more examples need to be stored in memory.<br>
                    ◽ If the batch size is too large, it can lead to out-of-memory (OOM) errors, especially when training on GPUs with limited memory.<br>
                    ◽ On the other hand, using a smaller batch size can result in slower training and may require more iterations to converge.<br>
                    ◽ The optimal batch size depends on various factors, such as the model architecture, available hardware resources, and the specific problem being solved.<br>
                    ◽ It's common to experiment with different batch sizes to find a balance between training speed and memory efficiency.<br>
                    ◽ If you encounter OOM errors, try reducing the batch size until the training can proceed without memory issues.<br>
                    ◽ Keep in mind that the batch size also affects the model's generalization and convergence behavior, so it's important to monitor the model's performance while adjusting the batch size.
                    """
                )
            with gr.Tab("Info - Learning Rate & Schedulers"):
                                gr.Markdown(
                    f"""
                    ### 🟩 <u>Learning Rate</u><br>
                    ◽ **1e-6:** Very small learning rate, slow but stable learning.<br>
                    ◽ **5e-6:** Small learning rate, slow but stable learning.<br>
                    ◽ **1e-5:** Moderate learning rate, balanced between stability and convergence speed.<br>
                    ◽ **5e-5:** Higher learning rate, faster convergence but potential instability.<br>
                    ◽ **1e-4:** High learning rate, faster convergence but increased risk of instability.<br>
                    ◽ **5e-4:** Very high learning rate, fast convergence but higher risk of instability.<br>
                    ◽ **1e-3:** Extremely high learning rate, very fast convergence but high risk of instability.<br><br>
                    The optimal learning rate depends on the model architecture, dataset, and other hyperparameters.

                    ### 🟩 <u>Learning Rate Schedulers</u><br>
                    ◽ **None:** Learning rate remains unchanged throughout the training. <br>
                    ◽ **Step:** Decreases the learning rate by a factor at specified intervals, offering a straightforward method for controlling learning rate decay.<br>
                    ◽ **MultiStep:** Decreases the learning rate by a factor at specified milestones, useful for gradually reducing the learning rate during training.<br>
                    ◽ **Exponential:** Decreases the learning rate exponentially over epochs, providing a simple decay mechanism for training.<br>
                    ◽ **CosineAnnealing:** Adjusts the learning rate following a cosine annealing schedule, facilitating smoother optimization and potentially better convergence.<br>
                    ◽ **Reduce On Plateau:** Reduces the learning rate when a monitored metric has stopped improving, helping to fine-tune learning rates during training.<br>
                    ◽ **Cyclic:** Cycles the learning rate between two boundary values, allowing for dynamic learning rate schedules during training.<br>
                    ◽ **One Cycle (disabled):** Sets the learning rate using a cyclical learning rate policy and varies momentum inversely, potentially accelerating training and improving performance.<br>
                    ◽ **Cosine Annealing Warm Restarts:** Adjusts the learning rate following a cosine annealing schedule with warm restarts, offering a balance between exploration and exploitation during optimization.<br><br>
                    The choice of learning rate scheduler depends on the specific characteristics of the model, dataset, and training objectives.
                    """
                )
            with gr.Tab("Info - Optimizers"):
                                gr.Markdown(
                    f"""
                    ### 🟩 <u>Optimizers</u><br>
                    ◽ **SGD with Momentum:** Good for convergence, requires careful tuning of the learning rate and momentum.<br>
                    ◽ **RMSprop:** Adaptively adjusts learning rates, good for varying gradient magnitudes.<br>
                    ◽ **Adam:** Widely used, performs well with default parameters.<br>
                    ◽ **Adagrad:** Good for sparse data, adapts learning rates based on historical gradients.<br>
                    ◽ **AdamW:** Similar to Adam but with decoupled weight decay for better regularization.<br><br>
                    When picking an optimizer, you need to experiment with different options to find the one that achieves the best performance and convergence for your task.<br><br>
                    AdamW has been set to default as its a sensible pick that will handle most needs.
                    """
                )
            with gr.Tab("Info - Epochs"):
                                gr.Markdown(
                    f"""
                    ### 🟩 <u>Epochs</u><br>
                    ◽ An epoch represents a single pass through the entire training dataset during the training process.<br>
                    ◽ In each epoch, the model sees and learns from all the training examples once.<br>
                    ◽ The number of epochs determines how many times the model will iterate over the complete training dataset.<br>
                    ◽ Increasing the number of epochs allows the model to learn more from the data and potentially improve its performance.<br>
                    ◽ However, training for too many epochs can lead to overfitting, where the model becomes too specialized to the training data and fails to generalize well to unseen data.<br>
                    ◽ The optimal number of epochs depends on factors such as the complexity of the problem, the size of the dataset, and the model's capacity.<br>
                    ◽ It's common to use techniques like early stopping or validation monitoring to determine the appropriate number of epochs.<br>
                    ◽ Early stopping involves monitoring the model's performance on a validation set and stopping the training when the performance starts to degrade, indicating potential overfitting.<br>
                    ◽ Validation monitoring involves evaluating the model's performance on a separate validation set after each epoch and selecting the model checkpoint that performs best on the validation set.<br>
                    ◽ It's recommended to start with a reasonable number of epochs and monitor the model's performance to determine if more epochs are needed or if early stopping should be applied
                    """
                )
            with gr.Tab("Info - Max Audio"):
                                gr.Markdown(
                    f"""
                    ### 🟩 <u>Max Permitted Audio Size (in seconds)</u><br>
                    ◽ The max permitted audio size determines the maximum duration of audio files that can be used as input to the model during training and inference.<br>
                    ◽ It is specified in seconds and represents the longest audio clip that the model can process.<br>
                    ◽ Limiting the audio size helps to control the memory usage and computational requirements of the model.<br>
                    ◽ If the audio files in your dataset vary in length, it's important to consider the max permitted audio size to ensure that all files can be processed effectively.<br>
                    ◽ Audio files longer than the max permitted size will typically be truncated or split into smaller segments before being fed into the model.<br>
                    ◽ The choice of max permitted audio size depends on the specific requirements of your application and the available hardware resources.<br>
                    ◽ Increasing the max permitted audio size allows the model to handle longer audio clips but also increases the memory consumption and computational burden.<br>
                    ◽ If you encounter memory issues or slow processing times, you may need to reduce the max permitted audio size to find a suitable balance.<br>
                    ◽ It's important to consider the nature of your audio data and choose a max permitted audio size that captures the relevant information while being computationally feasible.
                    """
                )

                
#######################
#### GRADIO STEP 3 ####
#######################
            
        with gr.Tab("✅ Step 3 - Testing"):
            with gr.Tab("Testing"):
                with gr.Row():
                    with gr.Column() as col1:
                        xtts_checkpoint = gr.Dropdown(
                            [str(file) for file in xtts_checkpoint_files],
                            label="XTTS checkpoint path (best_model.pth):",
                            value="",
                            allow_custom_value=True,
                        )

                        xtts_config = gr.Dropdown(
                            [str(file) for file in xtts_config_files],
                            label="XTTS config path (config.json):",
                            value="",
                            allow_custom_value=True,
                        )

                        xtts_vocab = gr.Dropdown(
                            [str(file) for file in xtts_vocab_files],
                            label="XTTS vocab path (vocab.json):",
                            value="",
                            allow_custom_value=True,
                        )
                        progress_load = gr.Label(
                            label="Progress:"
                        )
                        load_btn = gr.Button(value="Step 3 - Load Fine-tuned XTTS model")

                    with gr.Column() as col2:
                        with gr.Row():
                            # Gather the voice files
                            available_speaker_audios = get_available_voices()

                            # Create Dropdown for speaker reference audio
                            speaker_reference_audio = gr.Dropdown(
                                available_speaker_audios,
                                label="Speaker reference audio:",
                                value="",  # Set the default value if needed
                                allow_custom_value=True,  # Allow custom values
                                scale=1,
                            )
                            speaker_name_input_testing = gr.Textbox(
                                label="Project Name (Refresh Dropdowns on change)",
                                value="personsname",
                                visible=True,
                                scale=1,
                            )

                        with gr.Row():
                            tts_language = gr.Dropdown(
                                label="Language",
                                value="en",
                                choices=[
                                    "en",
                                    "es",
                                    "fr",
                                    "de",
                                    "it",
                                    "pt",
                                    "pl",
                                    "tr",
                                    "ru",
                                    "nl",
                                    "cs",
                                    "ar",
                                    "zh",
                                    "hu",
                                    "ko",
                                    "ja",
                                ]
                            )
                            # Create refresh button
                            refresh_button = create_refresh_button(
                                [xtts_checkpoint, xtts_config, xtts_vocab, speaker_reference_audio,
                                 speaker_name_input_testing],
                                [
                                    lambda speaker_name: {"choices": find_best_models(out_path, speaker_name=speaker_name), "value": ""},
                                    lambda speaker_name: {"choices": find_jsons(out_path, "config.json", speaker_name=speaker_name), "value": ""},
                                    lambda speaker_name: {"choices": find_jsons(out_path, "vocab.json", speaker_name=speaker_name),"value": ""},
                                    lambda speaker_name: {"choices": get_available_voices(speaker_name=speaker_name),"value": ""},
                                ],
                                elem_class="refresh-button-class"
                            )
                        tts_text = gr.Textbox(
                            label="Input Text:",
                            value="I've just fine tuned a text to speech language model and this is how it sounds. If it doesn't sound right, I will try a different Speaker Reference Audio file.",
                            lines=5,
                        )
                        tts_btn = gr.Button(value="Step 4 - Inference (Generate TTS)")

                with gr.Row():
                    progress_gen = gr.Label(
                        label="Progress:"
                    )
                with gr.Row():
                    tts_output_audio = gr.Audio(label="TTS Generated Speech.")
                    reference_audio = gr.Audio(label="Speaker Reference Audio Sample.")
                        
            with gr.Tab("Testing Instructions"):
                gr.Markdown(
                    f"""
                    ### ✅ <u>Testing</u><br>
                    ### 🟥 <u>Important Note</u>
                    ◽ If you dont believe the dropdown lists are populated correctly, you can use the "Refresh Dropdowns" button to search the finetuning folder path and update the lists. You can also manually edit the path to the listed files.<br>
                    ◽ You can use any config.json or vocab.json, as long as they match the model version you trained on. Only the XTTS checkpoint matters.<br>
                    ◽ Upon successful processing of your speech data, expect to find multiple speaker reference files if your input consisted of over 3 minutes of speech.<br>
                    ◽ Audio clips shorter than 7 seconds are automatically excluded from the reference list, as they generally do not provide sufficient length for effective TTS generation.<br>
                    ◽ Should you find a lower number of reference files than anticipated, it may be due to the initial segmentation performed by Whisper (Step 1), which can occasionally result in clips that are too brief for our criteria or indeed too long. In such cases, consider manually segmenting your audio to give Whisper a better chance at generating training data and repeating the process.<br>
                    ### 🟦 <u>What you need to do</u>
                    ◽ **Click** the **Refresh Dropdowns** button to correctly populate the **Speaker Reference Audio** with all the available WAV samples.<br>
                    ◽ The model is now trained and you are at the testing stage. Hopefully all the dropdowns should be pre-populated now.<br>
                    ◽ You need to <span style="color: #3366ff;">Load Fine-tuned XTTS model</span> and then select your <span style="color: #3366ff;">Speaker Reference Audio</span>. You can choose various <span style="color: #3366ff;">Speaker Reference Audios</span> to see which works best.<br>
                    ◽ All the <span style="color: #3366ff;">Speaker Reference Audios</span> in the dropdown are ones that are <span style="color: #3366ff;">8 seconds</span> long or more. You can use one of these later for your voice sample in All Talk, so remember the one you like.<br>
                    ◽ Some <span style="color: #3366ff;">Speaker Reference Audios</span> are a bit long and may cause issues generating the TTS, so try different ones.<br>
                    ◽ Once you are happy, move to the next tab for instruction on cleaning out the training data and how to copy your newly trained model over your current model.<br>
                    ### 🟨 <u>What this step is doing</u>
                    ◽ Its loading the finetuned model into memory, loading a voice sample and generating TTS, so that you can test out how well fine tuning worked.
                    """
                )                        
                        

        with gr.Tab("🔜 What to do next"):
            gr.Markdown(
                f"""
                ### 🔜 <u>What to do next</u><br>
                ### 🟦 <u>What you need to do</u>
                You have a few options below:<br>
                ◽ **Compact &amp; move model:** This will compress the raw finetuned model and move it, along with any large enough wav files created to the folder name of your choice.<br>
                ◽ **Delete Generated Training data:** This will delete the finetuned model and any other training data generated during this process. You will more than likely want to do this after you have Compacted &amp; moved the model.<br>
                ◽ **Delete original voice samples:** This will delete the voice samples you placed in put-voice-samples-in-here (if you no longer have a use for them).<br>
                ◽ **Note:** Your sample wav files will be copied into the model folder. You will need to **MANUALLY** copy/move the WAV file(s) you want to use to the AllTalk **voices** folder, to make them available within the interface.<br>
                ### 🟨 <u>Clearing up disk space</u>
                ◽ If you are not going to train anything again, you can delete the whisper model from inside of your huggingface cache (3GB approx) <br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◽ **Linux:** <span style="color: #3366ff;">~/.cache/huggingface/hub/(folder-here)</span><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◽ **Windows:** <span style="color: #3366ff;">C:&bsol;users&lpar;your-username&rpar;&bsol;.cache&bsol;huggingface&bsol;hub&bsol;(folder-here)</span>.<br>
                """
            )
            final_progress_data = gr.Label(
                label="Progress:"
            )
            with gr.Row():
                xtts_checkpoint_copy = gr.Dropdown(
                    [str(file) for file in xtts_checkpoint_files],
                    label="XTTS checkpoint path (Select the model you want to copy over):",
                    value="",
                    allow_custom_value=True,
                    scale=2,
                )
                speaker_name_input_export = gr.Textbox(
                    label="Project Name (Refresh Dropdowns on change)",
                    value="personsname",
                    visible=True,
                    scale=1,
                )
                # Create refresh button
                refresh_button = create_refresh_button_next(
                    [xtts_checkpoint_copy,speaker_name_input_export],
                    [
                        lambda speaker_name: {"choices": find_best_models(out_path, speaker_name), "value": ""},
                    ],
                    elem_class="refresh-button-class")
            with gr.Row():
                overwrite_existing = gr.Dropdown(value="Do not overwrite existing files", choices=["Overwrite existing files", "Do not overwrite existing files"], label="File Overwrite Options",)
                folder_path = gr.Textbox(label="Enter a new folder name (will be sub the models folder)", lines=1, value="mycustomfolder")
                compact_custom_btn = gr.Button(value="Compact and move model to a folder name of your choosing")
            with gr.Row():
                gr.Textbox(value="This will DELETE your training data and the raw finetuned model from /finetune/tmp-trn", scale=2, show_label=False, interactive=False)
                delete_training_btn = gr.Button(value="Delete generated training data")
            with gr.Row():
                gr.Textbox(value="This will DELETE your original voice samples from /finetune/put-voice-samples-in-here/.", scale=2, show_label=False, interactive=False)
                delete_voicesamples_btn = gr.Button(value="Delete original voice samples")

            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[
                    lang,
                    whisper_model,
                    max_sample_length,
                    eval_split_number,
                    speaker_name_input,
                    create_bpe_tokenizer
                ],
                outputs=[
                    progress_data,
                    train_csv,
                    eval_csv,
                    speaker_name_input_training
                ],
            )


            train_btn.click(
                fn=train_model,
                inputs=[
                    lang,
                    train_csv,
                    eval_csv,
                    learning_rates,
                    model_to_train, 
                    num_epochs,
                    batch_size,
                    grad_acumm,
                    max_audio_length,
                    speaker_name_input_training,
                    continue_run,
                    disable_shared_memory,
                    learning_rate_scheduler,
                    optimizer,
                    num_workers,
                    warm_up
                ],
                outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint, speaker_reference_audio, speaker_name_input_testing],
            )
            
            load_btn.click(
                fn=load_model,
                inputs=[
                    xtts_checkpoint,
                    xtts_config,
                    xtts_vocab
                ],
                outputs=[progress_load],
            )

            tts_btn.click(
                fn=run_tts,
                inputs=[
                    tts_language,
                    tts_text,
                    speaker_reference_audio,
                ],
                outputs=[progress_gen, tts_output_audio, reference_audio],
            )    
            compact_custom_btn.click(
                fn=compact_custom_model,
                inputs=[xtts_checkpoint_copy, folder_path, overwrite_existing],
                outputs=[final_progress_data],
            )
            delete_training_btn.click(
                fn=delete_training_data,
                outputs=[final_progress_data],
            )
            delete_voicesamples_btn.click(
                fn=delete_voice_sample_contents,
                outputs=[final_progress_data],
            )
            model_to_train.change(basemodel_or_finetunedmodel_choice, model_to_train, None)