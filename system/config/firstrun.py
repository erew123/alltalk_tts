import json
import os
import time
import shutil
import zipfile
import requests
from tqdm import tqdm
from inputimeout import inputimeout, TimeoutOccurred
from pathlib import Path

this_dir = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
# File paths
config_path = this_dir / 'confignew.json'
tts_engines_path = this_dir / 'system/tts_engines/tts_engines.json'
new_config_template_path = this_dir / 'system/config/confignew.json'

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=str(dest_path))
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

def setup_piper():
    os.makedirs(this_dir / 'models/piper', exist_ok=True)
    os.makedirs(this_dir / 'models/xtts', exist_ok=True)
    download_file("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx?download=true",
                  this_dir / "models/piper/en_US-ljspeech-high.onnx")
    download_file("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json?download=true.json",
                  this_dir / "models/piper/en_US-ljspeech-high.onnx.json")

def setup_vits():
    os.makedirs(this_dir / 'models/vits', exist_ok=True)
    os.makedirs(this_dir / 'models/xtts', exist_ok=True)
    zip_path = this_dir / 'models/vits/vits.zip'
    download_file("https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--vctk--vits.zip", zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(this_dir / 'models/vits')
    os.remove(zip_path)

def setup_xtts():
    os.makedirs(this_dir / 'models/xtts/xttsv2_2.0.3', exist_ok=True)
    base_url = "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/"
    files = [
        "LICENSE.txt",
        "README.md",
        "config.json",
        "model.pth",
        "dvae.pth",
        "mel_stats.pth",
        "speakers_xtts.pth",
        "vocab.json"
    ]
    for file in files:
        download_file(base_url + file + "?download=true", this_dir / f"models/xtts/xttsv2_2.0.3/{file}")



def update_tts_engines(engine, model):
    with open(tts_engines_path, 'r') as file:
        tts_engines = json.load(file)
    
    tts_engines['engine_loaded'] = engine
    tts_engines['selected_model'] = model
    
    with open(tts_engines_path, 'w') as file:
        json.dump(tts_engines, file, indent=4)

def set_firstrun_model_false():
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    config['firstrun_model'] = False
    
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

def load_config(file_path):
    with open(file_path, "r") as config_file:
        config = json.load(config_file)
    return config

def warning_mesasge():
    print(f"[{branding}TTS]")
    print(f"[{branding}TTS] \033[93mIf you have you have UPGRADED from v1 ensure you have re-installed\033[0m")
    print(f"[{branding}TTS] \033[93mthe requirements. Otherwise you will get failures and errors!\033[0m")
    print(f"[{branding}TTS] \033[93mOn Linux ignore the \033[0m'sparse_attn requires a torch version' \033[93mand\033[0m")
    print(f"[{branding}TTS] \033[0m'using untested triton version' \033[93mmessages.\033[0m")
    print(f"[{branding}TTS]")

# Load confignew.json
with open(config_path, 'r') as file:
    config = json.load(file)

branding = config['branding']

# Check if the config file is version 1
if "ip_address" in config:
    print(f"[{branding}TTS] Detected version 1 configuration file.")
    print(f"[{branding}TTS] Upgrading to version 2 configuration file...")

    # Backup the old configuration file
    backup_path = config_path + ".backup"
    shutil.copyfile(config_path, backup_path)

    # Copy the new configuration template to replace the old config file
    shutil.copyfile(new_config_template_path, config_path)

    print(f"[{branding}TTS] Configuration file upgraded successfully.")
    
    # Reload the new config file
    with open(config_path, 'r') as file:
        config = json.load(file)

# Check if firstrun_model is true
if config.get('firstrun_model', False):
    print(f"[{branding}TTS]")
    print(f"[{branding}TTS] \033[92mThis is the first time startup. Please download a start TTS model.\033[0m")
    print(f"[{branding}TTS]")
    
    # List of available models
    models = [
        {"name": "piper", "model": "piper"},
        {"name": "vits", "model": "tts_models--en--vctk--vits"},
        {"name": "xtts", "model": "xttsv2_2.0.3"},
    ]
    
    # Display models to the user
    print(f"[{branding}TTS]    \033[94mAvailable First Time Start-up models:\033[0m")
    print(f"[{branding}TTS]")
    for idx, model in enumerate(models):
        print(f"[{branding}TTS]    \033[93m{idx + 1}. \033[94m{model['name']} - {model['model']}\033[0m")
    print(f"[{branding}TTS]    \033[93m{len(models) + 1}.\033[94m I have my own models already\033[0m")
    print(f"[{branding}TTS]")
    print(f"[{branding}TTS]    \033[94mIn \033[91m60 seconds\033[0m \033[94ma Piper model will be \033[91mdownloaded automatically.\033[0m")
    print(f"[{branding}TTS]")
    
    # Auto-select model after 30 seconds
    selected_model = models[0]  # Default to the first model (piper)
    try:
        user_choice = inputimeout(prompt=f"[{branding}TTS]    \033[92mEnter your choice 1-4: \033[0m ", timeout=60)
    except TimeoutOccurred:
        user_choice = None

    if user_choice is None:
        print(f"[{branding}TTS]")
        print(f"[{branding}TTS] No input received. Proceeding with the default model (piper).")
    elif user_choice.lower() == 'own' or (user_choice.isdigit() and int(user_choice) == len(models) + 1):
        print(f"[{branding}TTS]")
        print(f"[{branding}TTS] If you have your own XTTS models, please move them into the")
        print(f"[{branding}TTS] \033[93m/models/xtts/\033[0m folder")
        print(f"[{branding}TTS]")
        os.makedirs('models/piper', exist_ok=True)
        os.makedirs('models/vits', exist_ok=True)
        os.makedirs('models/xtts/', exist_ok=True)
        update_tts_engines('piper', 'piper')
        set_firstrun_model_false()
        warning_mesasge()
        exit()

    if user_choice is not None and user_choice.isdigit() and 1 <= int(user_choice) <= len(models):
        selected_model = models[int(user_choice) - 1]
        warning_mesasge()

    if user_choice is None or user_choice.lower() != 'own':
        if selected_model['name'] == 'piper':
            setup_piper()
            update_tts_engines('piper', 'piper')
        elif selected_model['name'] == 'vits':
            setup_vits()
            update_tts_engines('vits', 'vits - tts_models--en--vctk--vits')
        elif selected_model['name'] == 'xtts':
            setup_xtts()
            update_tts_engines('xtts', 'xtts - xttsv2_2.0.3')

        print(f"[{branding}TTS] {selected_model['name']} model downloaded and configuration updated successfully.")
        # Set firstrun_model to false
        set_firstrun_model_false()
        warning_mesasge()
        exit()
else:
    exit()
