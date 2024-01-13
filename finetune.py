import argparse
import os
import sys
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
import pandas
import glob
import json
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel  
from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners


# STARTUP VARIABLES 
this_dir = Path(__file__).parent.resolve()
audio_folder = this_dir / "finetune" / "put-voice-samples-in-here"
out_path = this_dir / "finetune" / "tmp-trn"
progress = 0
theme = gr.themes.Default()
refresh_symbol = '🔄'
os.environ['TRAINER_TELEMETRY'] = '0'

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

def format_audio_list(target_language, whisper_model, out_path, gradio_progress=progress):
    audio_files = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith(('.mp3', '.flac', '.wav'))]
    buffer=0.2
    eval_percentage=0.15
    speaker_name="coqui"
    audio_total_size = 0
    os.makedirs(out_path, exist_ok=True)
    temp_folder = os.path.join(out_path, "temp")  # Update with your folder name
    os.makedirs(temp_folder, exist_ok=True)
    print("[FINETUNE] \033[94mPart of AllTalk\033[0m https://github.com/erew123/alltalk_tts/")
    print("[FINETUNE] \033[94mCoqui Public Model License\033[0m")
    print("[FINETUNE] \033[94mhttps://coqui.ai/cpml.txt\033[0m")
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
    asr_model = WhisperModel(whisper_model, device=device, compute_type="float16")

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

def train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path, max_audio_length=255995):
    #  Logging parameters
    RUN_NAME = "XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Set here the path that the checkpoints will be saved. Default: ./training/
    OUT_PATH = os.path.join(output_path, "training")
    print("[FINETUNE] \033[94mStarting Step 2\033[0m - Fine-tuning the XTTS Encoder")
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
        print("[FINETUNE] Starting finetuning on Base Model")
        TOKENIZER_FILE = str(this_dir / base_path / model_path / "vocab.json")
        XTTS_CHECKPOINT = str(this_dir / base_path / model_path / "model.pth")
        XTTS_CONFIG_FILE = str(this_dir / base_path / model_path / "config.json")
        DVAE_CHECKPOINT = str(this_dir / base_path / model_path / "dvae.pth")
        MEL_NORM_FILE = str(this_dir / base_path / model_path / "mel_stats.pth")
    else:
        # FINETUNED XTTS model checkpoints for fine-tuning.
        print("[FINETUNE] Starting finetuning on Existing Finetuned Model")
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
        num_loader_workers=8,
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
        lr=5e-06,  # learning rate
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

XTTS_MODEL = None
def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("[FINETUNE] \033[94mStarting Step 3\033[0m Loading XTTS model!")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
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

def find_models(directory, extension):
    """Find files with a specific extension in the given directory."""
    return [file for file in Path(directory).rglob(f"*.{extension}")]

def find_jsons(directory, filename):
    """Find files with a specific filename in the given directory."""
    return list(Path(directory).rglob(filename))

# Your main directory
main_directory = Path(this_dir) / "finetune" / "tmp-trn"
# XTTS checkpoint files (*.pth)
xtts_checkpoint_files = find_models(main_directory, "pth")
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

def compact_model():
    this_dir = Path(__file__).parent.resolve()

    best_model_path_str = find_latest_best_model(this_dir / "finetune" / "tmp-trn" / "training")

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

    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    # Define the target directory
    target_dir = this_dir / "models" / "trainedmodel"

    # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save the modified checkpoint in the target directory
    torch.save(checkpoint, target_dir / "model.pth")

    # Specify the files you want to copy
    files_to_copy = ["vocab.json", "config.json", "dvae.pth", "mel_stats.pth"]

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
    
    print("[FINETUNE] Model copied to '/models/trainedmodel/'")
    return "Model copied to '/models/trainedmodel/'"


def compact_lastfinetuned_model():
    this_dir = Path(__file__).parent.resolve()

    best_model_path_str = find_latest_best_model(this_dir / "finetune" / "tmp-trn" / "training")

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

    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    # Define the target directory
    target_dir = this_dir / "models" / "lastfinetuned"

    # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save the modified checkpoint in the target directory
    torch.save(checkpoint, target_dir / "model.pth")

    # Specify the files you want to copy
    files_to_copy = ["vocab.json", "config.json", "dvae.pth", "mel_stats.pth"]

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


def compact_legacy_model():
    this_dir = Path(__file__).parent.resolve()

    best_model_path_str = os.path.join(this_dir, "finetune", "best_model.pth")

    # Check if the best model file exists
    if not os.path.exists(best_model_path_str):
        print("[FINETUNE] No model called best_model.pth was found in /finetune/")
        return "No model called best_model.pth was found in /finetune/"

    # Convert model_path_str to Path
    best_model_path = Path(best_model_path_str)

    # Attempt to load the model
    try:
        checkpoint = torch.load(best_model_path, map_location=torch.device("cpu"))
    except Exception as e:
        print("[FINETUNE] Error loading checkpoint:", e)
        raise

    del checkpoint["optimizer"]

    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    # Define the target directory
    target_dir = this_dir / "finetune"

    # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save the modified checkpoint in the target directory
    torch.save(checkpoint, target_dir / "model.pth")

    print("[FINETUNE] model.pth created in '/finetune/'")
    return "model.pth created in '/finetune/'"


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


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_class, interactive=True):
    """
    Copied from https://github.com/AUTOMATIC1111/stable-diffusion-webui
    """
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = gr.Button(refresh_symbol, elem_classes=elem_class, interactive=interactive, scale=0)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )

    return refresh_button


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

    with gr.Blocks(theme=theme) as demo:
        with gr.Tab("Information/Guide"):

            gr.Markdown(
                f"""
                ## <u>Finetuning Information</u><br>
                ### 🟥 <u>Important Note</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - <span style="color: #3366ff;">finetune.py</span> needs to be run from the <span style="color: #3366ff;">/alltalk_tts/</span> folder. Don't move the location of this script.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Have you run AllTalk at least once? It needs to have downloaded+updated the voice model, before we can finetune it.
                ### 🟦 <u>What you need to run finetuning</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - An Nvidia GPU. Tested on Windows with extended shared VRAM and training used about 16GB's total (which worked on a 12GB card).
                #### &nbsp;&nbsp;&nbsp;&nbsp; - If you have multiple Nvidia GPU's in your system, please see this [important note](https://github.com/erew123/alltalk_tts#-i-have-multiple-gpus-and-i-have-problems-running-finetuning).
                #### &nbsp;&nbsp;&nbsp;&nbsp; - I have not been able to test this on a GPU with less than 12GB of VRAM, so cannot say if it will work or how that would affect performance.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - <span style="color: red;">Version 11.8</span> of Nvidia cuBLAS and cuDNN (guide below). Only 11.8 of cuBLAS and cuDNN work for this process currently.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Minimum <span style="color: red;">18GB</span> free disk space (most of it is used temporarily).
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Some decent quality audio, multiple files if you like. Minimum of 2 minutes and Ive tested up to 20 minutes of audio.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - There is no major need to chop down your audio files into small slices as Step 1 will do that for you automatically and prepare the training set. Ive been testing with 5 minute long clips.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - This process will need access to all your GPU and VRAM, so close any other software that's using your GPU currently.
                ### 🟨 <u>Setting up cuBLAS and cuDNN <span style="color: red;">Version 11.8</span></u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - If you have the <span style="color: #3366ff;;">Nvidia CUDA Toolkit Version 11.8</span> installed and can type <span style="color: #3366ff;;">nvcc --version</span> at the command prompt/terminal and it reports <span style="color: #00a000;">Cuda compilation tools, release 11.8</span> you should be good to go. 
                #### &nbsp;&nbsp;&nbsp;&nbsp; - If you dont have the toolkit installed, the idea is just to install the smallest bit possible and this will not affect or impact other things on your system.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - You will need to download the Nvidia Cuda Toolkit 11.8<span style="color: #3366ff;"> network</span> install from [here](https://developer.nvidia.com/cuda-11-8-0-download-archive) 
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1) Run the installer and select <span style="color: #3366ff;">Custom Advanced</span> Uncheck <span style="color: #3366ff;">everything</span> at the top then expand <span style="color: #3366ff;">CUDA</span>, <span style="color: #3366ff;">Development</span> > <span style="color: #3366ff;">Compiler</span> > and select <span style="color: #3366ff;;">nvcc</span> then expand <span style="color: #3366ff;;">Libraries</span> and select <span style="color: #3366ff;;">CUBLAS</span>.
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2) Back at the top of <span style="color: #3366ff;">CUDA</span>, expand <span style="color: #3366ff;">Runtime</span> > <span style="color: #3366ff;">Libraries</span> and select <span style="color: #3366ff;">CUBLAS</span>. Click <span style="color: #3366ff;;">Next</span>, accept the default path (taking a note of its location) and let the install run. 
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3) You should be able to drop to your terminal or command prompt and type <span style="color: #3366ff;">nvcc --version</span> and have it report <span style="color: #00a000;">Cuda compilation tools, release 11.8</span>. If it does you are good to go. If it doesn't > Step 4.
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4) Linux users, you can temporarily add these paths on your current terminal window with (you may need to confirm these are correct for your flavour of Linux):
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #3366ff;">export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64&colon;&dollar;&lbrace;LD_LIBRARY_PATH&colon;&plus;&colon;&dollar;&lbrace;LD_LIBRARY_PATH&rbrace;&rbrace;</span> (Add it to your ~/.bashrc if you want this to be permanent)
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #3366ff;">export LD_LIBRARY_PATH=/usr/local/cuda-11.8/bin</span>
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Windows users need the add the following to the PATH environment variable. Start menu and search for "Environment Variables" or "Edit the system environment variables.". 
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Find and select the "Path" variable, then click on the "Edit...". Click on the "New" button and add:
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #3366ff;">C:&bsol;Program Files&bsol;NVIDIA GPU Computing Toolkit&bsol;CUDA&bsol;v11.8&bsol;bin.</span>
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5) Once you have these set correctly, you should be able to open a new command prompt/terminal and <span style="color: #3366ff;">nvcc --version</span> at the command prompt/terminal, resulting in <span style="color: #00a000;">Cuda compilation tools, release 11.8</span>.
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6) If the nvcc command doesn't work OR it reports a version different from 11.8, finetuning wont work, so you will to double check your environment variables and get them working correctly.
                """
            )

#######################
#### GRADIO STEP 1 ####
#######################
        with gr.Tab("Step 1 - Preparing Audio/Generating the dataset"):

            gr.Markdown(
                f"""
                ## <u>STEP 1 - Preparing Audio/Generating the dataset</u><br>
                ### 🟦 <u>What you need to do</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Please read Coqui's guide on what makes a good dataset [here](https://docs.coqui.ai/en/latest/what_makes_a_good_dataset.html#what-makes-a-good-dataset)
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Place your audio files in <span style="color: #3366ff;">{str(audio_folder)}</span>                
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Your audio samples can be in the format <span style="color: #3366ff;">mp3, wav,</span> or <span style="color: #3366ff;">flac.</span>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - You will need a minimum of <span style="color: #3366ff;">2 minutes</span> of audio in either one or multiple audio files. Very small sample files cause errors, so I would suggest 30 seconds and longer samples. 
                #### &nbsp;&nbsp;&nbsp;&nbsp; - When you have completed Steps 1, 2, and 3, you are welcome to delete your samples from "put-voice-samples-in-here".
                #### &nbsp;&nbsp;&nbsp;&nbsp; - FYI Anecdotal evidence suggests that the Whisper 2 model may yield superior results in audio splitting and dataset creation.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - If this step is failing, it will be worth running the diagnostics with atsetup and confirming you have cu118 or cu112 listed against your torch and torchaudio.<br>
                ### 🟨 <u>What this step is doing</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - With step one, we are going to be stripping your audio file(s) into smaller files, using Whisper to find spoken words/sentences, compile that into excel sheets of training data, ready for finetuning the model on Step 2.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Whilst you can choose multiple Whisper models, its best only to use the 1 model as each one is about 3GB in size and will download to your local huggingface cache on first-time use. If and when you have completed training, you wish to delete this 3GB model from your system, you are welcome to do so.
                ### 🟩 <u>How long will this take?</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - First time, it needs to download the Whisper model which is 3GB. After that a few minutes on an average 3-4 year old system.
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
                    value="large-v3",
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
            progress_data = gr.Label(
                label="Progress:"
            )
            logs = gr.Textbox(
                label="Logs:",
                interactive=False,
            )
            demo.load(read_logs, None, logs, every=1)

            prompt_compute_btn = gr.Button(value="Step 1 - Create dataset")
        
            def preprocess_dataset(language,  whisper_model, out_path, progress=gr.Progress(track_tqdm=True)):
                clear_gpu_cache()
                test_for_audio_files = [file for file in os.listdir(audio_folder) if any(file.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac'])]
                if not test_for_audio_files:
                    return "I cannot find any mp3, wav or flac files in the folder called 'put-voice-samples-in-here'", "", ""
                else:
                    try:
                        train_meta, eval_meta, audio_total_size = format_audio_list(target_language=language, whisper_model=whisper_model ,out_path=out_path, gradio_progress=progress)
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
        with gr.Tab("Step 2 - Fine-tuning the XTTS Encoder"):

            gr.Markdown(
                f"""
                ## <u>STEP 2 - Fine-tuning the XTTS Encoder</u><br>
                ### 🟦 <u>What you need to do</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - The <span style="color: #3366ff;">Train CSV</span> and <span style="color: #3366ff;">Eval CSV</span> should already be populated. If not, just go back to Step 1 and click the button again.             
                #### &nbsp;&nbsp;&nbsp;&nbsp; - If for any reason it crashed, close the app and re-start the finetuning and kick it off again, clicking through setp 1 (it doesnt need to regenerate the data)
                #### &nbsp;&nbsp;&nbsp;&nbsp; - The settings below are factory and should be ok.<br>
                ### 🟨 <u>What this step is doing</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Very simply put, it's taking all our wav files generated in Step 1, along with our recorded speech in out excel documents and its training the model on that voice.
                ### 🟩 <u>How long will this take?</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - On a RTX 4070 with factory settings (as below) and 20 minutes of audio, this took 20 minutes. Again, it will vary by system and how much audio you are throwing at it. You have time to go make a coffee.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Look at your command prompt/terminal window if you want to see what it is doing.
                """
            )
            with gr.Row():
                train_csv = gr.Textbox(
                    label="Train CSV:",
                )
                eval_csv = gr.Textbox(
                    label="Eval CSV:",
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
                    minimum=2,
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
                ##### If you have an existing finetuned model in <span style="color: #3366ff;">/models/trainedmodel/</span> you will have an option to select <span style="color: #3366ff;">Existing finetuned model</span>, allowing you to further train it. Otherwise, you will only have Base Model as an option.
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

            def train_model(language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
                clear_gpu_cache()
                if not train_csv or not eval_csv:
                    return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
                try:
                    # convert seconds to waveform frames
                    max_audio_length = int(max_audio_length * 22050)
                    config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=str(output_path), max_audio_length=max_audio_length)
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}", "", "", "", ""

                # copy original files to avoid parameters changes issues
                os.system(f"cp {config_path} {exp_path}")
                os.system(f"cp {vocab_file} {exp_path}")

                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
                print("[FINETUNE] Model training done. Move to Step 3")
                clear_gpu_cache()
                return "Model training done. Move to Step 3", config_path, vocab_file, ft_xtts_checkpoint, speaker_wav
            
#######################
#### GRADIO STEP 3 ####
#######################
            
        with gr.Tab("Step 3 - Inference/Testing"):
            gr.Markdown(
                f"""
                ## <u>STEP 3 -  Inference/Testing</u><br>
                ### 🟥 <u>Important Note</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - This step will error if you are using TTS version 0.22.0. Please re-run  <span style="color: #3366ff;">pip install -r requirements_finetune.txt</span> if it errors loading the model.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - If you dont see multiple speaker reference files and you used more than 3 minutes of speech, try refreshing the page as it may not have loaded them correctly.
                ### 🟦 <u>What you need to do</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - The model is now trained and you are at the testing stage. Hopefully all the dropdowns should be pre-populated now.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - You need to <span style="color: #3366ff;">Load Fine-tuned XTTS model</span> and then select your <span style="color: #3366ff;">Speaker Reference Audio</span>. You can choose various <span style="color: #3366ff;">Speaker Reference Audios</span> to see which works best.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - All the <span style="color: #3366ff;">Speaker Reference Audios</span> in the dropdown are ones that are <span style="color: #3366ff;">8 seconds</span> long or more. You can use one of these later for your voice sample in All Talk, so remember the one you like.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Some <span style="color: #3366ff;">Speaker Reference Audios</span> are a bit long and may cause issues generating the TTS, so try different ones.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Once you are happy, move to the next tab for instruction on cleaning out the training data and how to copy your newly trained model over your current model.
                ### 🟨 <u>What this step is doing</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Its loading the finetuned model into memory, loading a voice sample and generating TTS, so that you can test out how well fine tuning worked.
                """
            )

            with gr.Row():
                with gr.Column() as col1:
                    xtts_checkpoint = gr.Dropdown(
                        [str(file) for file in xtts_checkpoint_files],
                        label="XTTS checkpoint path:",
                        value="",
                        allow_custom_value=True,
                    )

                    xtts_config = gr.Dropdown(
                        [str(file) for file in xtts_config_files],
                        label="XTTS config path:",
                        value="",
                        allow_custom_value=True,
                    )

                    xtts_vocab = gr.Dropdown(
                        [str(file) for file in xtts_vocab_files],
                        label="XTTS vocab path:",
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
                        # Create refresh button
                        refresh_button = create_refresh_button(
                            speaker_reference_audio,
                            lambda: None,  # Specify your refresh logic if needed
                            lambda: {"choices": get_available_voices(), "value": ""},  # Update choices and value
                            "refresh-button",
                        )

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

        with gr.Tab("What to do next"):
            gr.Markdown(
                f"""
                ## <u>What to do next</u><br>
                ### 🟦 <u>What you need to do</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - You have a few options below:
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 🔹 Compact &amp; move model. This will compress the raw finetuned model and move it, along with any large enough wav files created to one of two locations (please read text next to the button).<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You can use the wav files in that location as sample voices with your model. Models moved to <span style="color: #3366ff;">/models/trainedmodel/</span> will be loadable within Text-gen-webui on the <span style="color: #3366ff;">XTTSv2 FT</span> loader.
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 🔹 Delete Generated Training data. This will delete the finetuned model and any other training data generated during this process. You will more than likely want to do this after you have Compacted &amp; moved the model.
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 🔹 Delete original voice samples. This will delete the voice samples you placed in put-voice-samples-in-here (if you no longer have a use for them).
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 🔹 Compress a legacy finetuned model. For any models that were generated pre the current version of finetuning, you can still compact them down.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - Instructions on using the finetuned model in Text-generation-webui and also further tuning this model you just created are on <span style="color: #3366ff;">OPTION A - Compact and move model to '/trainedmodel/'</span>.
                ### 🟨 <u>Clearing up disk space</u>
                #### &nbsp;&nbsp;&nbsp;&nbsp; - If you are not going to train anything again, you can delete the whisper model from inside of your huggingface cache (3GB approx) 
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Linux <span style="color: #3366ff;">~/.cache/huggingface/hub/(folder-here)</span>
                #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Windows <span style="color: #3366ff;">C:&bsol;users&lpar;your-username&rpar;&bsol;.cache&bsol;huggingface&bsol;hub&bsol;(folder-here)</span>.
                #### &nbsp;&nbsp;&nbsp;&nbsp; - You can also uninstall the Nvidia CUDA 11.8 Toolkit if you wish and remove your environment variable entries.
                """
            )
            final_progress_data = gr.Label(
                label="Progress:"
            )
            with gr.Row():
                compact_btn = gr.Button(value="OPTION A - Compact and move model to '/trainedmodel/'")
                gr.Markdown(
                f"""
                ##### This will compact the model and move it, along with the reference wav's to <span style="color: #3366ff;">/models/trainedmodel/</span> and will <span style="color: red;">OVERWRITE</span> any model in that location. This model will become available to load in Text-gen-webui with the <span style="color: #3366ff;">XTTSv2 FT</span> loader. It will also be detected by Finetune as a model that can be further trained (Reload finetune.py).
                """
                )
            with gr.Row():
                compact_lastfinetuned_btn = gr.Button(value="OPTION B - Compact and move model to '/lastfinetuned/'")
                gr.Markdown(
                f"""
                ##### This will compact the model and move it, along with the reference wav's to <span style="color: #3366ff;">/models/lastfinetuned/</span> and will <span style="color: red;">OVERWRITE</span> any model in that location. This will just leave you a finetuned model that you can do with as you please. It will <span style="color: red;">NOT</span> be available in Text-gen-webui unless you copy it over one of the other model loader locations.
                """
                )
            with gr.Row():
                delete_training_btn = gr.Button(value="Delete generated training data")
                gr.Markdown(
                f"""
                ##### This will <span style="color: red;">DELETE</span> your training data and the raw finetuned model from <span style="color: #3366ff;">/finetune/tmp-trn/</span>.
                """
                )
            with gr.Row():
                delete_voicesamples_btn = gr.Button(value="Delete original voice samples")
                gr.Markdown(
                f"""
                ##### This will <span style="color: red;">DELETE</span> your original voice samples from <span style="color: #3366ff;">/finetune/put-voice-samples-in-here/</span>.
                """
                )
            with gr.Row():
                compact_legacy_btn = gr.Button(value="Compact a legacy finetuned model")
                gr.Markdown(
                f"""
                ##### This will compact your old finetuned models from 5GB to 1.9GB. Place your 5GB <span style="color: #3366ff;">base_model.pth</span> file in <span style="color: #3366ff;">/finetune/</span> (rename your file to <span style="color: #3366ff;">base_model.pth</span> if necessary). This will generate you a <span style="color: #3366ff;">model.pth</span> file.
                """
                )

            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[
                    lang,
                    whisper_model,
                    out_path,
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
                outputs=[final_progress_data],
            )
            compact_lastfinetuned_btn.click(
                fn=compact_lastfinetuned_model,
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
            compact_legacy_btn.click(
                fn=compact_legacy_model,
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

