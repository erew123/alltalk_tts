import argparse
import os
import sys
import platform
import site
import tempfile
import signal
import gradio as gr
import torch
import torchaudio
import traceback
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import random
import gc
import time
import shutil
import psutil
import pandas
import glob
import json
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel  
# Use a local Tokenizer to resolve Japanese support
# from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners
from system.ft_tokenizer.tokenizer import multilingual_cleaners
import importlib.metadata as metadata
from packaging import version

# STARTUP VARIABLES 
this_dir = Path(__file__).parent.resolve()
audio_folder = this_dir / "finetune" / "put-voice-samples-in-here"
out_path = this_dir / "finetune" / "tmp-trn"
progress = 0
theme = gr.themes.Default()
refresh_symbol = '🔄'
os.environ['TRAINER_TELEMETRY'] = '0'
pfc_status = "pass"

# Define the path to the modeldownload config file file
modeldownload_config_file_path = this_dir / "modeldownload.json"

# Check if the JSON file exists
if modeldownload_config_file_path.exists():
    with open(modeldownload_config_file_path, "r") as config_file:
        settings = json.load(config_file)
    # Extract settings from the loaded JSON
    base_path = Path(settings.get("base_path", ""))
    model_path = Path(settings.get("model_path", ""))
    base_model_path = Path(settings.get("model_path", ""))
    files_to_download = settings.get("files_to_download", {})
else:
    # Default settings if the JSON file doesn't exist or is empty
    print("[FINETUNE]  \033[91mWarning\033[0m modeldownload.json is missing. Please run this script in the /alltalk_tts/ folder")
    sys.exit(1)

##################################################
#### Check to see if a finetuned model exists ####
##################################################
# Set the path to the directory
trained_model_directory = this_dir / "models" / "trainedmodel"
# Check if the directory "trainedmodel" exists
finetuned_model = trained_model_directory.exists()
# If the directory exists, check for the existence of the required files
# If true, this will add a extra option in the Gradio interface for loading Xttsv2 FT
if finetuned_model:
    required_files = ["model.pth", "config.json", "vocab.json", "mel_stats.pth", "dvae.pth"]
    finetuned_model = all((trained_model_directory / file).exists() for file in required_files)
basemodel_or_finetunedmodel = True

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

def generate_cuda_markdown():
    global pfc_status
    cuda_status, cuda_icon, cuda_home = test_cuda()
    file_name = 'cublas64_11.*' if platform.system() == "Windows" else 'libcublas.so.11*'
    found_paths = find_files_in_path_with_wildcard(file_name)
    if found_paths:
        found_paths_str = ' '.join(found_paths)
        found_path_icon = '✅'
    else:
        found_paths_str = "cublas64_11 is not accessible."
        found_path_icon = '❌'
        pfc_status = "fail"  # Update global status
        # Check if 'cu118' or 'cu121' is in the PyTorch version string
    pytorch_version = torch.__version__
    if 'cu118' in pytorch_version or 'cu121' in pytorch_version:
        pytorch_cuda_version_status = ''
        pytorch_icon = '✅'
    else:
        pytorch_cuda_version_status = 'Pytorch CUDA version problem '
        pytorch_icon = '❌'
        pfc_status = "fail"  # Update global status
    cuda_markdown = f"""
    ### 🟨 <u>CUDA Information</u><br>
    &nbsp;&nbsp;&nbsp;&nbsp; {found_path_icon} **Cublas64_11 found:** {found_paths_str}  
    &nbsp;&nbsp;&nbsp;&nbsp; {pytorch_icon} **CUDA_HOME path:** {cuda_home}
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


def check_base_model(base_model_path, files_to_download):
    global pfc_status
    # Assuming files_to_download is a dict with keys as filenames
    base_model_files = list(files_to_download.keys())
    missing_files = []
    # Check if all base model files exist
    for file in base_model_files:
        file_path = this_dir / base_path / model_path / file
        if not file_path.exists():
            missing_files.append(file)
            pfc_status = "fail"
    return len(missing_files) == 0

# Assuming base_model_path and files_to_download are set from the JSON config as shown above
base_model_detected = check_base_model(base_model_path, files_to_download)

def generate_base_model_markdown(base_model_detected):
    global pfc_status
    base_model_status = 'Base model detected' if base_model_detected else 'Base model not detected'
    base_model_icon = '✅' if base_model_detected else '❌'
    base_model_markdown = f"""
    ### ⬛ <u>XTTS Base Model Detection</u>
    &nbsp;&nbsp;&nbsp;&nbsp; {base_model_icon} **Base XTTS Model Status:** {base_model_status}
    """
    return base_model_markdown

def check_tts_version(required_version="0.22.0"):
    global pfc_status
    try:
        # Get the installed version of TTS
        installed_version = metadata.version("tts")
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

def format_audio_list(target_language, whisper_model, out_path, eval_split_number, speaker_name_input, gradio_progress=progress):
    pfc_check_fail()
    audio_files = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith(('.mp3', '.flac', '.wav'))]
    buffer=0.2
    eval_percentage = eval_split_number / 100.0
    speaker_name=speaker_name_input
    audio_total_size = 0
    os.makedirs(out_path, exist_ok=True)
    temp_folder = os.path.join(out_path, "temp")  # Update with your folder name
    os.makedirs(temp_folder, exist_ok=True)
    print("[FINETUNE] \033[94mPart of AllTalk\033[0m https://github.com/erew123/alltalk_tts/")
    print("[FINETUNE] \033[94mCoqui Public Model License\033[0m")
    print("[FINETUNE] \033[94mhttps://coqui.ai/cpml.txt\033[0m")
    print(f"[FINETUNE] \033[94mWhisper model: \033[92m{whisper_model} \033[94mLanguage: \033[92m{target_language} \033[94mEvaluation data percentage: \033[92m{eval_split_number}%\033[0m")
    print("[FINETUNE] \033[94mStarting Step 1\033[0m - Preparing Audio/Generating the dataset")
    # Write the target language to lang.txt in the output directory
    lang_file_path = os.path.join(out_path, "lang.txt")

    # Check if lang.txt already exists and contains a different language
    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()

    if current_language != target_language:
        # Only update lang.txt if target language is different from the current language
        with open(lang_file_path, 'w', encoding='utf-8') as lang_file:
            lang_file.write(target_language + '\n')
        print("[FINETUNE] Updated lang.txt with the target language.")
    else:
        print("[FINETUNE] The existing language matches the target language")

    # Loading Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[FINETUNE] Loading Whisper Model:", whisper_model)
    print("[FINETUNE] Model will be downloaded if its not available, which will take a few minutes.")
    asr_model = WhisperModel(whisper_model, device=device, compute_type="float32")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    existing_metadata = {'train': None, 'eval': None}
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")

    if os.path.exists(train_metadata_path):
        existing_metadata['train'] = pandas.read_csv(train_metadata_path, sep="|")
        print("[FINETUNE] Existing training metadata found and loaded.")

    if os.path.exists(eval_metadata_path):
        existing_metadata['eval'] = pandas.read_csv(eval_metadata_path, sep="|")
        print("[FINETUNE] Existing evaluation metadata found and loaded.")

    for idx, audio_path in tqdm(enumerate(audio_files)):
        if isinstance(audio_path, str):
            audio_file_name_without_ext, _ = os.path.splitext(os.path.basename(audio_path))
            # If it's a string, it's already the path to the file
            audio_path_name = audio_path
        elif hasattr(audio_path, 'read'):
            # If it has a 'read' attribute, treat it as a file-like object
            # and use a temporary file to save its content
            audio_file_name_without_ext, _ = os.path.splitext(os.path.basename(audio_path.name))
            audio_path_name = create_temporary_file(temp_folder)
            with open(audio_path, 'rb') as original_file:
                file_content = original_file.read()
            with open(audio_path_name, 'wb') as temp_file:
                    temp_file.write(file_content)

        # Create a temporary file path within the new folder
        temp_audio_path = create_temporary_file(temp_folder)

        try:
            if isinstance(audio_path, str):
                audio_path_name = audio_path
            elif hasattr(audio_path, 'name'):
                audio_path_name = audio_path.name
            else:
                raise ValueError(f"Unsupported audio_path type: {type(audio_path)}")
        except Exception as e:
            print("[FINETUNE] Error reading original file: {e}")
            # Handle the error or raise it if needed
        print("[FINETUNE] Current working file:", audio_path_name)
        try:
            # Copy the audio content
            time.sleep(0.5)  # Introduce a small delay
            shutil.copy2(audio_path_name, temp_audio_path)
        except Exception as e:
            print("[FINETUNE] Error copying file: {e}")
            # Handle the error or raise it if needed

        # Load the temporary audio file
        wav, sr = torchaudio.load(temp_audio_path, format="wav")
        wav = torch.as_tensor(wav).clone().detach().t().to(torch.float32), sr

        prefix_check = f"wavs/{audio_file_name_without_ext}_"

        # Check both training and evaluation metadata for an entry that starts with the file name.
        skip_processing = False

        for key in ['train', 'eval']:
            if existing_metadata[key] is not None:
                mask = existing_metadata[key]['audio_file'].str.startswith(prefix_check)

                if mask.any():
                    print(f"[FINETUNE] Segments from {audio_file_name_without_ext} have been previously processed; skipping...")
                    skip_processing = True
                    break

        # If we found that we've already processed this file before, continue to the next iteration.
        if skip_processing:
            continue

        wav, sr = torchaudio.load(audio_path)
        # stereo to mono if needed
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(audio_path, vad_filter=True, word_timestamps=True, language=target_language)
        segments = list(segments)
        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        # added all segments words in a unique list
        words_list = []
        for _, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        # process each word
        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start = word.start
                # If it is the first sentence, add buffer or get the beginning of the file
                if word_idx == 0:
                    sentence_start = max(sentence_start - buffer, 0)  # Add buffer to the sentence start
                else:
                    # get the previous sentence end
                    previous_word_end = words_list[word_idx - 1].end
                    # add buffer or get the silence middle between the previous sentence and the current one
                    sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start) / 2)

                sentence = word.word
                first_word = False
            else:
                sentence += word.word

            if word.word[-1] in ["!", ".", "?"]:
                sentence = sentence[1:]
                # Expand number and abbreviations plus normalization
                sentence = multilingual_cleaners(sentence, target_language)
                audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))

                audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

                # Check for the next word's existence
                if word_idx + 1 < len(words_list):
                    next_word_start = words_list[word_idx + 1].start
                else:
                    # If don't have more words it means that it is the last sentence then use the audio len as next word start
                    next_word_start = (wav.shape[0] - 1) / sr

                # Average the current word end and next word start
                word_end = min((word.end + next_word_start) / 2, word.end + buffer)

                absolute_path = os.path.join(out_path, audio_file)
                os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
                i += 1
                first_word = True

                audio = wav[int(sr * sentence_start):int(sr * word_end)].unsqueeze(0)
                # if the audio is too short, ignore it (i.e., < 0.33 seconds)
                if audio.size(-1) >= sr / 3:
                    torchaudio.save(
                        absolute_path,
                        audio,
                        sr
                    )
                else:
                    continue

                metadata["audio_file"].append(audio_file)
                metadata["text"].append(sentence)
                metadata["speaker_name"].append(speaker_name)

        os.remove(temp_audio_path)

    if os.path.exists(train_metadata_path) and os.path.exists(eval_metadata_path):
        existing_train_df = existing_metadata['train']
        existing_eval_df = existing_metadata['eval']
        audio_total_size = 121
    else:
        existing_train_df = pandas.DataFrame(columns=["audio_file", "text", "speaker_name"])
        existing_eval_df = pandas.DataFrame(columns=["audio_file", "text", "speaker_name"])

    new_data_df = pandas.DataFrame(metadata)

    combined_train_df = pandas.concat([existing_train_df, new_data_df], ignore_index=True).drop_duplicates().reset_index(drop=True)

    combined_train_df_shuffled = combined_train_df.sample(frac=1)
    num_val_samples = int(len(combined_train_df_shuffled) * eval_percentage)

    final_eval_set = combined_train_df_shuffled[:num_val_samples]
    final_training_set = combined_train_df_shuffled[num_val_samples:]

    final_training_set.sort_values('audio_file').to_csv(train_metadata_path, sep='|', index=False)
    final_eval_set.sort_values('audio_file').to_csv(eval_metadata_path, sep='|', index=False)

    # deallocate VRAM and RAM
    del asr_model, final_eval_set, final_training_set, new_data_df, existing_metadata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    existing_train_df = None
    existing_eval_df = None
    print("[FINETUNE] Train CSV:", train_metadata_path)
    print("[FINETUNE] Eval CSV:", eval_metadata_path)
    print("[FINETUNE] Audio Total:", audio_total_size)
    return train_metadata_path, eval_metadata_path, audio_total_size

######################
#### STEP 2 BITS #####
######################

from trainer import Trainer, TrainerArgs

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

def train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, learning_rate, output_path, max_audio_length=255995):
    pfc_check_fail()
    #  Logging parameters
    RUN_NAME = "XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Set here the path that the checkpoints will be saved. Default: ./training/
    OUT_PATH = os.path.join(output_path, "training")
    print("[FINETUNE] \033[94mStarting Step 2\033[0m - Fine-tuning the XTTS Encoder")
    print(f"[FINETUNE] \033[94mLanguage: \033[92m{language} \033[94mEpochs: \033[92m{num_epochs} \033[94mBatch size: \033[92m{batch_size}\033[0m \033[94mGrad accumulation steps: \033[92m{grad_acumm}\033[0m")
    print(f"[FINETUNE] \033[94mTraining   : \033[92m{train_csv}\033[0m")
    print(f"[FINETUNE] \033[94mEvaluation : \033[92m{eval_csv}\033[0m")
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

    if basemodel_or_finetunedmodel:
        # BASE XTTS model checkpoints for fine-tuning.
        print("[FINETUNE] Starting finetuning on \033[92mBase Model\033[0m")
        TOKENIZER_FILE = str(this_dir / base_path / model_path / "vocab.json")
        XTTS_CHECKPOINT = str(this_dir / base_path / model_path / "model.pth")
        XTTS_CONFIG_FILE = str(this_dir / base_path / model_path / "config.json")
        DVAE_CHECKPOINT = str(this_dir / base_path / model_path / "dvae.pth")
        MEL_NORM_FILE = str(this_dir / base_path / model_path / "mel_stats.pth")
    else:
        # FINETUNED XTTS model checkpoints for fine-tuning.
        print("[FINETUNE] Starting finetuning on \033[92mExisting Finetuned Model\033[0m")
        TOKENIZER_FILE = str(this_dir / base_path / "trainedmodel" / "vocab.json")
        XTTS_CHECKPOINT = str(this_dir / base_path / "trainedmodel" / "model.pth")
        XTTS_CONFIG_FILE = str(this_dir / base_path / "trainedmodel" / "config.json")
        DVAE_CHECKPOINT = str(this_dir / base_path / "trainedmodel" / "dvae.pth")
        MEL_NORM_FILE = str(this_dir / base_path / "trainedmodel" / "mel_stats.pth")

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
    number_of_workers = 8
    if language == "ja":
        number_of_workers = 0

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
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=learning_rate,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)
    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
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
    return XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, trainer_out_path, speaker_ref

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
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    xtts_speakers_pth = find_a_speaker_file(this_dir / "models")
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("[FINETUNE] \033[94mStarting Step 3\033[0m Loading XTTS model!")
    print(xtts_checkpoint)
    print(xtts_vocab)
    print(xtts_speakers_pth)
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False, speaker_file_path=xtts_speakers_pth)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("[FINETUNE] Model Loaded!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None
    speaker_audio_file = str(speaker_audio_file)
    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
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

def get_available_voices(minimum_size_kb=1200):
    voice_files = [
        voice for voice in Path(f"{this_dir}/finetune/tmp-trn/wavs").glob("*.wav")
        if voice.stat().st_size > minimum_size_kb * 1200  # Convert KB to bytes
    ]
    return sorted([str(file) for file in voice_files])  # Return full path as string

def find_best_models(directory):
    """Find files named 'best_model.pth' in the given directory."""
    return [str(file) for file in Path(directory).rglob("best_model.pth")]

def find_models(directory, extension):
    """Find files with a specific extension in the given directory."""
    return [str(file) for file in Path(directory).rglob(f"*.{extension}")]

def find_jsons(directory, filename):
    """Find files with a specific filename in the given directory."""
    return [str(file) for file in Path(directory).rglob(filename)]

# Your main directory
main_directory = Path(this_dir) / "finetune" / "tmp-trn"
# XTTS checkpoint files (best_model.pth)
xtts_checkpoint_files = find_best_models(main_directory)
# XTTS config files (config.json)
xtts_config_files = find_jsons(main_directory, "config.json")
# XTTS vocab files (vocab.json)
xtts_vocab_files = find_jsons(main_directory, "vocab.json")

##########################
#### STEP 4 AND OTHER ####
##########################

def find_latest_best_model(folder_path):
    search_path = folder_path / "XTTS_FT-*" / "best_model.pth"
    files = glob.glob(str(search_path), recursive=True)
    latest_file = max(files, key=os.path.getctime, default=None)
    return latest_file

def compact_model(xtts_checkpoint_copy):
    this_dir = Path(__file__).parent.resolve()
    print("THIS DIR:", this_dir)
    best_model_path_str = str(xtts_checkpoint_copy)  # Convert to string
    print("best_model_path_str", best_model_path_str)
    
    # Check if the best model file exists
    if not best_model_path_str:
        print("[FINETUNE] No trained model was found.")
        return "No trained model was found."
    
    print(f"[FINETUNE] Best model path: {best_model_path_str}")
    
    # Attempt to load the model
    try:
        checkpoint = torch.load(best_model_path_str, map_location=torch.device("cpu"))
        print(f"[FINETUNE] Checkpoint loaded: {best_model_path_str}")
    except Exception as e:
        print("[FINETUNE] Error loading checkpoint:", e)
        raise
    
    # Define the target directory
    target_dir = this_dir / "models" / "trainedmodel"
    
    # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    del checkpoint["optimizer"]

    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]
    
    # Save the modified checkpoint in the target directory
    torch.save(checkpoint, str(target_dir / "model.pth"))  # Convert to string

    # Specify the files you want to copy
    files_to_copy = ["vocab.json", "config.json", "speakers_xtts.pth", "mel_stats.pth", "dvae.pth"]
    
    for file_name in files_to_copy:
        src_path = this_dir / base_path / base_model_path / file_name
        dest_path = target_dir / file_name
        shutil.copy(str(src_path), str(dest_path))  # Convert to string
    
    source_wavs_dir = this_dir / "finetune" / "tmp-trn" / "wavs"
    target_wavs_dir = target_dir / "wavs"
    target_wavs_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate through files in the source directory
    for file_path in source_wavs_dir.iterdir():
        # Check if it's a file and larger than 1000 KB
        if file_path.is_file() and file_path.stat().st_size > 1000 * 1024:
            # Copy the file to the target directory
            shutil.copy(str(file_path), str(target_wavs_dir / file_path.name))  # Convert to string
    
    print("[FINETUNE] Model copied to '/models/trainedmodel/'")
    return "Model copied to '/models/trainedmodel/'"


def compact_lastfinetuned_model(xtts_checkpoint_copy):
    this_dir = Path(__file__).parent.resolve()

    best_model_path_str = xtts_checkpoint_copy
    print(f"[FINETUNE] Best model path: {best_model_path_str}")

    # Check if the best model file exists
    if best_model_path_str is None:
        print("[FINETUNE] No trained model was found.")
        return "No trained model was found."

    # Convert model_path_str to Path
    best_model_path = Path(best_model_path_str)

    # Attempt to load the model
    try:
        checkpoint = torch.load(best_model_path, map_location=torch.device("cpu"))
    except Exception as e:
        print("[FINETUNE] Error loading checkpoint:", e)
        raise

    del checkpoint["optimizer"]

    # Define the target directory
    target_dir = this_dir / "models" / "lastfinetuned"

    # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    # Save the modified checkpoint in the target directory
    torch.save(checkpoint, target_dir / "model.pth")

    # Specify the files you want to copy
    files_to_copy = ["vocab.json", "config.json", "speakers_xtts.pth", "mel_stats.pth", "dvae.pth",]

    for file_name in files_to_copy:
        src_path = this_dir / base_path / base_model_path / file_name
        dest_path = target_dir / file_name
        shutil.copy(str(src_path), str(dest_path))

    source_wavs_dir = this_dir / "finetune" / "tmp-trn" / "wavs"
    target_wavs_dir = target_dir / "wavs"
    target_wavs_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through files in the source directory
    for file_path in source_wavs_dir.iterdir():
        # Check if it's a file and larger than 1000 KB
        if file_path.is_file() and file_path.stat().st_size > 1000 * 1024:
            # Copy the file to the target directory
            shutil.copy(str(file_path), str(target_wavs_dir / file_path.name))
    
    print("[FINETUNE] Model copied to '/models/lastfinetuned/'")
    return "Model copied to '/models/lastfinetuned/'"


def compact_custom_model(xtts_checkpoint_copy, folder_path):
    this_dir = Path(__file__).parent.resolve()

    best_model_path_str = xtts_checkpoint_copy
    print(f"[FINETUNE] Best model path: {best_model_path_str}")

    # Check if the best model file exists
    if best_model_path_str is None:
        print("[FINETUNE] No trained model was found.")
        return "No trained model was found."

    # Convert model_path_str to Path
    best_model_path = Path(best_model_path_str)

    # Attempt to load the model
    try:
        checkpoint = torch.load(best_model_path, map_location=torch.device("cpu"))
    except Exception as e:
        print("[FINETUNE] Error loading checkpoint:", e)
        raise

    del checkpoint["optimizer"]

    # Define the target directory
    target_dir = this_dir / "models" / folder_path

    # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    # Save the modified checkpoint in the target directory
    torch.save(checkpoint, target_dir / "model.pth")

    # Specify the files you want to copy
    files_to_copy = ["vocab.json", "config.json", "speakers_xtts.pth", "mel_stats.pth", "dvae.pth",]

    for file_name in files_to_copy:
        src_path = this_dir / base_path / base_model_path / file_name
        dest_path = target_dir / file_name
        shutil.copy(str(src_path), str(dest_path))

    source_wavs_dir = this_dir / "finetune" / "tmp-trn" / "wavs"
    target_wavs_dir = target_dir / "wavs"
    target_wavs_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through files in the source directory
    for file_path in source_wavs_dir.iterdir():
        # Check if it's a file and larger than 1000 KB
        if file_path.is_file() and file_path.stat().st_size > 1000 * 1024:
            # Copy the file to the target directory
            shutil.copy(str(file_path), str(target_wavs_dir / file_path.name))
    
    print("[FINETUNE] Model copied to '/models/",folder_path,"/")
    return f"Model copied to '/models/{folder_path}/'"


def delete_training_data():
    # Define the folder to be deleted
    folder_to_delete = Path(this_dir / "finetune" / "tmp-trn")

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

def delete_voice_sample_contents():
    # Define the folder to be cleared
    folder_to_clear = Path(this_dir / "finetune" / "put-voice-samples-in-here")

    # Check if the folder exists before clearing its contents
    if folder_to_clear.exists() and folder_to_clear.is_dir():
        # List all files and subdirectories in the folder
        for item in os.listdir(folder_to_clear):
            item_path = folder_to_clear / item
            if item_path.is_file():
                # If it's a file, remove it
                os.remove(item_path)
            elif item_path.is_dir():
                # If it's a subdirectory, remove it recursively
                shutil.rmtree(item_path)

        print(f"[FINETUNE] Contents of {folder_to_clear} deleted successfully.")
        return f"Contents of 'put-voice-samples-in-here deleted' successfully."
    else:
        print(f"[FINETUNE] Folder {folder_to_clear} does not exist.")
        return f"Folder 'put-voice-samples-in-here' does not exist."

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

def read_logs():
    sys.stdout.flush()
    with open(sys.stdout.log_file, "r") as f:
        return f.read()

def cleanup_before_exit(signum, frame):
    print("[FINETUNE] Received interrupt signal. Cleaning up and exiting...")
    # Perform cleanup operations here if necessary
    sys.exit(0)

def create_refresh_button(refresh_components, refresh_methods, elem_class, interactive=True):
    def refresh():
        updates = {}
        for component, method in zip(refresh_components, refresh_methods):
            args = method() if callable(method) else method
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
        inputs=[],
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

    with gr.Blocks(theme=theme, css=custom_css) as demo:
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
                ◽ Ensure you have started AllTalk normally at least once. You can start it again and it will download any missing files.<br>
                ◽ Check that there is an XTTS model within the models folder e.g. <span style="color: #3366ff;">/models/xttsv2_2.0.2/</span><br>
                ◽ The files required are "model.pth", "vocab.json", "config.json", "dvae.pth", "mel_stats.pth", "speakers_xtts.pth".
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

            gr.Markdown(
                f"""
                ### 📁 <u>Generating the dataset</u><br>
                ### 🟥 <u>Important Note - Windows - "UserWarning: huggingface_hub cache-system uses symlinks."</u>
                ◽ This error is caused by Huggingfaces download software downloading the Whisper model. If you get this error, please restart your Windows command prompt with "Run as Administrator" and restart finetuning.<br>
                ◽ This should only occur the 1st time it downloads the Whisper model. [Huggingface Reference here](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache#limitations)
                ### 🟥 <u>Important Note - Language support.</u>
                ◽ Although I have done my best to help automate this step, the Whisper model is not great with all languages and may fail to build your training data correctly. The Large-v3 may be better at certain languages.<br>
                ◽ You can find information about the Whisper model [here](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages) and you can find data about manually building training data [here](https://docs.coqui.ai/en/latest/formatting_your_dataset.html), as well as details below about the file structure this step performs.<br>
                ### 🟦 <u>What you need to do</u>
                ◽ Please read Coqui's guide on what makes a good dataset [here](https://docs.coqui.ai/en/latest/what_makes_a_good_dataset.html#what-makes-a-good-dataset)<br>
                ◽ Place your audio files in <span style="color: #3366ff;">{str(audio_folder)}</span>          
                ◽ Your audio samples can be in the format <span style="color: #3366ff;">mp3, wav,</span> or <span style="color: #3366ff;">flac.</span><br>
                ◽ You will need a minimum of <span style="color: #3366ff;">2 minutes</span> of audio in either one or multiple audio files. 5 to 10 minutes of audio would probably be better, allowing for more varied sample data to be generated.<br>
                ◽ Very small sample files cause errors, so I would recommend that the samples are at least 30 seconds and longer.<br>
                ◽ FYI Anecdotal evidence suggests that the Whisper 2 model may yield superior results in audio splitting and dataset creation.<br>
                ### 🟨 <u>What this step is doing</u><br>
                ◽ With step one, we are going to be stripping your audio file(s) into smaller files, using Whisper to find spoken words/sentences, compile that into excel sheets of training data, ready for Step 2.<br>
                ◽ Whisper is making a best effort to find spoken audio and break it down into smaller audio files. The content of these audio files is then transcribed into CSV fles (which you can edit in Excel or similar).<br>
                ◽ These files (audio and CSV) are used at the next step to train the model "this is what the audio sounds like and these are the words being spoken".<br>
                ◽ If you wish to manually look at the CSV files before running the next step, you are welcome to do so and edit them as necessary. The greater the accuracy of the text the better the training will be.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◽ <span style="color: #3366ff;">/alltalk_tts/finetune/tmp-trn/lang.txt</span> Contains a two digit langauage code e.g. `en`, `fr` etc.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◽ <span style="color: #3366ff;">/alltalk_tts/finetune/tmp-trn/metadata_train.csv</span> Contains the list of wavs to text that will be used. This is what the model is trained with.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◽ <span style="color: #3366ff;">/alltalk_tts/finetune/tmp-trn/metadata_eval.csv</span> Lists the evaluation wav to text, used to generate TTS while training and evaluate the quality of generated TTS to the original sample.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◽ <span style="color: #3366ff;">/alltalk_tts/finetune/tmp-trn/wavs/</span> These are the wavs split out from your original training samples and match the contents of the CSV files.<br>
                ◽ Whilst you can choose multiple Whisper models, its best only to use the 1 model as each one is about 3GB in size and will download to your local huggingface cache on first-time use. <br>
                ◽ If and when you have completed training, should you wish to delete this 3GB Whisper model from your system, you are welcome to do so.<br>
                ### 🟩 <u>How long will this take?</u><br>
                ◽ First time, it needs to download the Whisper model which is 3GB. After that a few minutes on an average 3-4 year old system.<br>
                """
            )
                 
            out_path = gr.Textbox(
                label="Output path (where data and checkpoints will be saved):",
                value=out_path,
                visible=False,
            )
            with gr.Row():
                whisper_model = gr.Dropdown(
                    label="Whisper Model",
                    value="large-v2",
                    choices=[
                        "large-v3",
                        "large-v2",
                        "large",
                        "medium",
                        "small"
                    ],
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
                )

                eval_split_number = gr.Number(
                    label="Evaluation data Split (the % to use for Evaluation data)",
                    value=15,  # Default value
                    minimum=5,  # Minimum value
                    maximum=95,  # Maximum value
                    step=1,  # Increment step
                )
                speaker_name_input = gr.Textbox(
                label="The name of the speaker/person you are training",
                value="personsname",
                visible=True,
            )
            progress_data = gr.Label(
                label="Progress:"
            )
            logs = gr.Textbox(
                label="Logs:",
                interactive=False,
            )
            demo.load(read_logs, None, logs, every=1)

            prompt_compute_btn = gr.Button(value="Step 1 - Create dataset")
        
            def preprocess_dataset(language, whisper_model, out_path, eval_split_number, speaker_name_input, progress=gr.Progress(track_tqdm=True)):
                clear_gpu_cache()
                test_for_audio_files = [file for file in os.listdir(audio_folder) if any(file.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac'])]
                if not test_for_audio_files:
                    return "I cannot find any mp3, wav or flac files in the folder called 'put-voice-samples-in-here'", "", ""
                else:
                    try:

                        train_meta, eval_meta, audio_total_size = format_audio_list(target_language=language, whisper_model=whisper_model, out_path=out_path, eval_split_number=eval_split_number, speaker_name_input=speaker_name_input, gradio_progress=progress)
                    except:
                        traceback.print_exc()
                        error = traceback.format_exc()
                        return f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}", "", ""

                clear_gpu_cache()

                # if audio total len is less than 2 minutes raise an error
                if audio_total_size < 120:
                    message = "The total duration of the audio file or files you provided, was less than 2 minutes in length. Please add more audio samples."
                    print("[FINETUNE] ", message)
                    return message, "", ""

                print("[FINETUNE] Dataset Generated. Move to Step 2")
                return "Dataset Generated. Move to Step 2", train_meta, eval_meta
            
#######################
#### GRADIO STEP 2 ####
#######################
        with gr.Tab("💻 Step 2 - Training"):
            with gr.Tab("Training"):
                gr.Markdown(
                    f"""
                    ### 💻 <u>Training</u><br>
                    ### 🟥 <u>Important Note - Language support.</u>
                    ◽ If this step is failing/erroring you may wish to check your training data was created correctly (Detailed in Step 1), confirming that wav files have been generated and your `metadata_train.csv` and `metadata_eval.csv` files have been populated.<br>             
                    ### 🟦 <u>What you need to do</u>
                    ◽ The <span style="color: #3366ff;">Train CSV</span> and <span style="color: #3366ff;">Eval CSV</span> should already be populated. If not, just go back to Step 1 and click "Create Dataset" again.<br>
                    ◽ The default settings below are the suggested settings for most purposes, however you may choose to alter them depending on your specific use case.<br>
                    ◽ There are no absolute correct settings for training. It will vary based on:<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- What you are training (A human voice, A cartoon voice, A new language entirely etc)<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- How much audio you have (you may want less or more eval or epochs)<br>
                    ◽ The key indicator of sufficient model training is whether it sounds right to you. Coqui suggests beginning with the base settings for training. If the resulting audio doesn't meet your expectations, additional training sessions may be necessary.<br>
                    ### 🟨 <u>What this step is doing</u><br>
                    ◽ Very simply put, it's taking all our wav files generated in Step 1, along with our recorded speech in out excel documents and its training the model on that voice e.g. listen to this audio file and this is what is being said in it, so learn to reproduce it.<br>
                    ### 🟩 <u>How long will this take?</u><br>
                    ◽ On a RTX 4070 with factory settings (as below) and 20 minutes of audio, this took 20 minutes. Again, it will vary by system and how much audio you are throwing at it. You have time to go make a coffee.<br>
                    ◽ Look at your command prompt/terminal window if you want to see what it is doing.<br>
                    """
                )
                with gr.Row():
                    train_csv = gr.Textbox(
                        label="Train CSV:",
                        scale=2,
                    )
                    eval_csv = gr.Textbox(
                        label="Eval CSV:",
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
                        maximum=32,
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
                with gr.Row():
                    model_to_train_choices = ["Base Model"]
                    if finetuned_model:
                        model_to_train_choices.append("Existing finetuned model")
                    model_to_train = gr.Radio(
                        choices=model_to_train_choices,
                        label="Which model do you want to train?",
                        value="Base Model"
                    )
                    gr.Markdown(
                    f"""
                    If you have an existing finetuned model in <span style="color: #3366ff;">/models/trainedmodel/</span> you will have an option to select <span style="color: #3366ff;">Existing finetuned model</span>, allowing you to further train it. Otherwise, you will only have Base Model as an option.
                    """
                    )

                progress_train = gr.Label(
                    label="Progress:"
                )
                logs_tts_train = gr.Textbox(
                    label="Logs:",
                    interactive=False,
                )
                demo.load(read_logs, None, logs_tts_train, every=1)
                train_btn = gr.Button(value="Step 2 - Run the training")

                def train_model(language, train_csv, eval_csv, learning_rates, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
                    clear_gpu_cache()
                    if not train_csv or not eval_csv:
                        return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
                    try:
                        # convert seconds to waveform frames
                        max_audio_length = int(max_audio_length * 22050)
                        learning_rate = float(learning_rates)  # Convert the learning rate value to a float
                        config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, learning_rate, output_path=str(output_path), max_audio_length=max_audio_length)
                    except:
                        traceback.print_exc()
                        error = traceback.format_exc()
                        return f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}", "", "", "", ""

                    # copy original files to avoid parameters changes issues
                    shutil.copy(config_path, exp_path)
                    shutil.copy(vocab_file, exp_path)

                    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
                    print("[FINETUNE] Model training done. Move to Step 3")
                    clear_gpu_cache()
                    return "Model training done. Move to Step 3", config_path, vocab_file, ft_xtts_checkpoint, speaker_wav
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
            with gr.Tab("Info - Learning Rate"):
                                gr.Markdown(
                    f"""
                    ### 🟩 <u>Learning Rate</u><br>
                    ◽ 1e-6: Very small learning rate, slow but stable learning.<br>
                    ◽ 5e-6: Small learning rate, slow but stable learning.<br>
                    ◽ 1e-5: Moderate learning rate, balanced between stability and convergence speed.<br>
                    ◽ 5e-5: Higher learning rate, faster convergence but potential instability.<br>
                    ◽ 1e-4: High learning rate, faster convergence but increased risk of instability.<br>
                    ◽ 5e-4: Very high learning rate, fast convergence but higher risk of instability.<br>
                    ◽ 1e-3: Extremely high learning rate, very fast convergence but high risk of instability.<br><br>
                    The optimal learning rate depends on the model architecture, dataset, and other hyperparameters.
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
                            [xtts_checkpoint, xtts_config, xtts_vocab, speaker_reference_audio],
                            [
                                lambda: {"choices": find_best_models(main_directory), "value": ""},
                                lambda: {"choices": find_jsons(main_directory, "config.json"), "value": ""},
                                lambda: {"choices": find_jsons(main_directory, "vocab.json"), "value": ""},
                                lambda: {"choices": get_available_voices(), "value": ""},
                            ],
                            elem_class="refresh-button-class"
                        )
                    tts_text = gr.Textbox(
                        label="Input Text:",
                        value="I've just fine tuned a text to speech language model and this is how it sounds. If it doesn't sound right, I will try a different Speaker Reference Audio file.",
                        lines=6,
                    )
                    tts_btn = gr.Button(value="Step 4 - Inference (Generate TTS)")

                with gr.Column() as col3:
                    progress_gen = gr.Label(
                        label="Progress:"
                    )
                    tts_output_audio = gr.Audio(label="TTS Generated Audio.")
                    reference_audio = gr.Audio(label="Speaker Reference Audio Used.")

        with gr.Tab("🔜 What to do next"):
            gr.Markdown(
                f"""
                ### 🔜 <u>What to do next</u><br>
                ### 🟦 <u>What you need to do</u>
                You have a few options below:<br>
                ◽ **Compact &amp; move model:** This will compress the raw finetuned model and move it, along with any large enough wav files created to one of two locations (please read text next to the button). You can use the wav files in that location as sample voices with your model. Models moved to <span style="color: #3366ff;">/models/trainedmodel/</span> will be loadable within Text-gen-webui on the <span style="color: #3366ff;">XTTSv2 FT</span> loader.<br>
                ◽ **Delete Generated Training data:** This will delete the finetuned model and any other training data generated during this process. You will more than likely want to do this after you have Compacted &amp; moved the model.<br>
                ◽ **Delete original voice samples:** This will delete the voice samples you placed in put-voice-samples-in-here (if you no longer have a use for them).<br>
                ◽ **Note:** Sample wav files will be copied into the model folder. You will need to MANUALLY copy/move the WAV file(s) you want to use to the AllTalk voices folder, to make them available within the interface.<br>
                ◽ **Note:** Text-generation-webui users, <span style="color: #3366ff;">OPTION A - Compact and move model to '/trainedmodel/'</span>. will make the model available for use within Text-gen-webui.<br>
                ### 🟩 <u>Using your Finetuned model as you main model</u>
                ◽ **Currently** only Text-gen-webui and SillyTavern have loaders built for loading models in the <span style="color: #3366ff;">'/finetuned/'</span> folder. So if you wish to use your finetuned model as elsewhere in AllTalk please copy it over your base model e.g. copy your model over the top of <span style="color: #3366ff;">/models/xttsv2_2.0.2/</span><br>
                ◽ If you wish to use the model you have just fintuned as your main model, you will need to MANUALLY copy it over the base model e.g copy it over <span style="color: #3366ff;">/models/xttsv2_2.0.2/</span> after you have used Option A or B.<br>
                ◽ Should you ever wish to revert your model back to the base model, you can delete the files in <span style="color: #3366ff;">/models/xttsv2_2.0.2/</span> and new copies will be re-downloaded when AllTalk starts up.<br>
                ### 🟨 <u>Clearing up disk space</u>
                ◽ If you are not going to train anything again, you can delete the whisper model from inside of your huggingface cache (3GB approx) <br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◽ **Linux:** <span style="color: #3366ff;">~/.cache/huggingface/hub/(folder-here)</span><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ◽ **Windows:** <span style="color: #3366ff;">C:&bsol;users&lpar;your-username&rpar;&bsol;.cache&bsol;huggingface&bsol;hub&bsol;(folder-here)</span>.<br>
                ◽ You can also uninstall the Nvidia CUDA 11.8 Toolkit if you wish and remove your environment variable entries.<br>
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
                    scale=3,
                )
                # Create refresh button
                refresh_button = create_refresh_button(
                    [xtts_checkpoint_copy,],
                    [
                        lambda: {"choices": find_best_models(main_directory), "value": ""},
                    ],
                    elem_class="refresh-button-class")
            with gr.Row():
                compact_btn = gr.Button(value="OPTION A - Compact and move model to '/trainedmodel/'")
                gr.Markdown(
                f"""
                This will compact the model and move it, along with the reference wav's to <span style="color: #3366ff;">/models/trainedmodel/</span> and will <span style="color: red;">OVERWRITE</span> any model in that location. This model will become available to load in Text-gen-webui with the <span style="color: #3366ff;">XTTSv2 FT</span> loader. It will also be detected by Finetune as a model that can be further trained (Reload finetune.py).
                """
                )
            with gr.Row():
                compact_lastfinetuned_btn = gr.Button(value="OPTION B - Compact and move model to '/lastfinetuned/'")
                gr.Markdown(
                f"""
                This will compact the model and move it, along with the reference wav's to <span style="color: #3366ff;">/models/lastfinetuned/</span> and will <span style="color: red;">OVERWRITE</span> any model in that location. This will just leave you a finetuned model that you can do with as you please. It will <span style="color: red;">NOT</span> be available in Text-gen-webui unless you copy it over one of the other model loader locations.
                """
                )
            with gr.Row():
                compact_custom_btn = gr.Button(value="OPTION C - Compact and move model to a folder name of your choosing")
                folder_path = gr.Textbox(label="Enter a new folder name (will be sub the models folder)", lines=1, value="mycustomfolder")
            with gr.Row():
                delete_training_btn = gr.Button(value="Delete generated training data")
                gr.Markdown(
                f"""
                This will <span style="color: red;">DELETE</span> your training data and the raw finetuned model from <span style="color: #3366ff;">/finetune/tmp-trn/</span>.
                """
                )
            with gr.Row():
                delete_voicesamples_btn = gr.Button(value="Delete original voice samples")
                gr.Markdown(
                f"""
                This will <span style="color: red;">DELETE</span> your original voice samples from <span style="color: #3366ff;">/finetune/put-voice-samples-in-here/</span>.
                """
                )

            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[
                    lang,
                    whisper_model,
                    out_path,
                    eval_split_number,
                    speaker_name_input,
                ],
                outputs=[
                    progress_data,
                    train_csv,
                    eval_csv,
                ],
            )


            train_btn.click(
                fn=train_model,
                inputs=[
                    lang,
                    train_csv,
                    eval_csv,
                    learning_rates,
                    num_epochs,
                    batch_size,
                    grad_acumm,
                    out_path,
                    max_audio_length,
                ],
                outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint, speaker_reference_audio],
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
            compact_btn.click(
                fn=compact_model,
                inputs=[xtts_checkpoint_copy],
                outputs=[final_progress_data],
            )
            compact_lastfinetuned_btn.click(
                fn=compact_lastfinetuned_model,
                inputs=[xtts_checkpoint_copy],
                outputs=[final_progress_data],
            )
            compact_custom_btn.click(
                fn=compact_custom_model,
                inputs=[xtts_checkpoint_copy, folder_path],
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

    demo.queue().launch(
        show_api=False,
        inbrowser=True,
        share=False,
        debug=False,
        server_port=7052,
        server_name="127.0.0.1",
    )

