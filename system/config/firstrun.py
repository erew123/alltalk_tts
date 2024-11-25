import json
import os
import shutil
import time
import sys
import zipfile
import requests
from tqdm import tqdm
from inputimeout import inputimeout, TimeoutOccurred
from pathlib import Path

# Setting the module search path:
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from config import AlltalkTTSEnginesConfig, AlltalkConfig

this_dir = Path(__file__).parent.resolve().parent.resolve().parent.resolve()

def is_running_in_colab():
    """Test for google colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_running_in_docker():
    """Test for google colab"""
    path = '/proc/self/cgroup'
    return os.path.exists('/.dockerenv') or (
        os.path.exists(path) and any('docker' in line for line in open(path))
    )

def download_file(url: str, dest_path: str, timeout: int = 30, retries: int = 3) -> bool:
    """
    Download a file from a URL to a specified destination path with progress bar.
    
    Args:
        url (str): URL of the file to download
        dest_path (str): Destination path where the file will be saved
        timeout (int, optional): Connection and read timeout in seconds. Defaults to 30.
        retries (int, optional): Number of retry attempts for failed downloads. Defaults to 3.
    """
    progress_bar = None
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True, 
                                 timeout=(timeout, timeout))
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            # Initialize progress bar
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, 
                              desc=f"Downloading {Path(dest_path).name} (Attempt {attempt + 1}/{retries})")
            Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'wb') as dwn_file:
                for data in response.iter_content(block_size):
                    if data:
                        dwn_file.write(data)
                        progress_bar.update(len(data))
            if progress_bar:
                progress_bar.close()
            if progress_bar.n != total_size and total_size > 0:
                raise requests.exceptions.RequestException(
                    f"Downloaded size ({progress_bar.n} bytes) does not match expected size ({total_size} bytes)"
                )    
            return True
        except requests.exceptions.RequestException as e:
            if progress_bar:
                progress_bar.close()
            if Path(dest_path).exists():
                Path(dest_path).unlink()
            if attempt == retries - 1:
                raise requests.exceptions.RequestException(
                    f"Failed to download {url} after {retries} attempts: {str(e)}"
                )
            print(f"Download attempt {attempt + 1}/{retries} failed: {str(e)}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            if progress_bar:
                progress_bar.close()
            raise Exception(f"Unexpected error downloading {url}: {str(e)}")
    return False

def setup_piper():
    """Download base piper files"""
    os.makedirs(this_dir / 'models/piper', exist_ok=True)
    os.makedirs(this_dir / 'models/xtts', exist_ok=True)
    download_file("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx?download=true",
                  this_dir / "models/piper/en_US-ljspeech-high.onnx")
    download_file("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json?download=true.json",
                  this_dir / "models/piper/en_US-ljspeech-high.onnx.json")

def setup_vits():
    """Download base vits files"""
    os.makedirs(this_dir / 'models/vits', exist_ok=True)
    os.makedirs(this_dir / 'models/xtts', exist_ok=True)
    zip_path = this_dir / 'models/vits/vits.zip'
    download_file("https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--vctk--vits.zip", zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(this_dir / 'models/vits')
    os.remove(zip_path)

def setup_xtts():
    """Download base xtts files"""
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
    for each_file in files:
        download_file(base_url + file + "?download=true", this_dir / f"models/xtts/xttsv2_2.0.3/{each_file}")

def update_tts_engines(engine):
    """Set engine to users choice that matches the download"""
    try:
        config_instance = AlltalkTTSEnginesConfig.get_instance()
        config_instance.change_engine(engine)
        config_instance.save()
    except Exception as e:
        print(f"[{branding}TTS] Error updating TTS engine configuration: {str(e)}")

def set_firstrun_model_false():
    try:
        # Get a fresh instance of the config
        config_fr = AlltalkConfig.get_instance()
        config_fr.firstrun_model = False

        # Force a flush to disk immediately
        config_fr.save()

        # Verify the save worked
        config_path = AlltalkConfig.default_config_path()
        with open(config_path, 'r', encoding="utf-8") as f:
            saved_config = json.load(f)
            if saved_config.get('firstrun_model', True):
                print(f"[{branding}TTS] Warning: Failed to save firstrun_model=False")
                # Try one more time with direct file write
                saved_config['firstrun_model'] = False
                with open(config_path, 'w', encoding="utf-8") as f:
                    json.dump(saved_config, f, indent=2)
    except Exception as e:
        print(f"[{branding}TTS] Error saving firstrun configuration: {str(e)}")

def warning_message():
    print(f"[{branding}TTS]")
    print(f"[{branding}TTS] \033[93mIf you have you have UPGRADED from v1 ensure you have re-installed\033[0m")
    print(f"[{branding}TTS] \033[93mthe requirements. Otherwise you will get failures and errors!\033[0m")
    print(f"[{branding}TTS] \033[93mOn Linux ignore the \033[0m'sparse_attn requires a torch version' \033[93mand\033[0m")
    print(f"[{branding}TTS] \033[0m'using untested triton version' \033[93mmessages.\033[0m")
    print(f"[{branding}TTS]")

# Load confignew.json (might be version 1 or 2):
with open(AlltalkConfig.default_config_path(), 'r', encoding="utf-8") as file:
    config_json = json.load(file)

branding = config_json['branding']

# Check if the config file is version 1
if "ip_address" in config_json:
    print(f"[{branding}TTS] Detected version 1 configuration file.")
    print(f"[{branding}TTS] Upgrading to version 2 configuration file...")

    # Backup the old configuration file
    backup_path = AlltalkConfig.default_config_path() + ".backup"
    shutil.copyfile(AlltalkConfig.default_config_path(), backup_path)

    # Copy the new configuration template to replace the old config file
    new_config_template_path = this_dir / 'system/config/confignew.json'
    shutil.copyfile(new_config_template_path, AlltalkConfig.default_config_path())

    print(f"[{branding}TTS] Configuration file upgraded successfully.")

config = AlltalkConfig.get_instance()
branding = config.branding

# Check if firstrun_model is true
if config.firstrun_model:
    # Check for Colab or Docker environment
    if is_running_in_colab() or is_running_in_docker():
        print(f"[{branding}TTS] Detected Colab/Docker environment - automatically selecting Piper")
        setup_piper()
        update_tts_engines('piper')
        set_firstrun_model_false()
        warning_message()
        exit()

    print(f"[{branding}TTS]")
    print(f"[{branding}TTS] \033[92mThis is the first time startup.. Please download a start TTS model. Other TTS engines\033[0m")
    print(f"[{branding}TTS] \033[92mand TTS models can be downloaded/managed in the Gradio Interface `TTS Engines Settings`\033[0m")
    print(f"[{branding}TTS] \033[92mtab after inital setup.\033[0m")
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

    # Auto-select model after 60 seconds
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
        update_tts_engines('piper')
        set_firstrun_model_false()
        warning_message()
        exit()

    if user_choice is not None and user_choice.isdigit() and 1 <= int(user_choice) <= len(models):
        selected_model = models[int(user_choice) - 1]
        warning_message()

    if user_choice is None or user_choice.lower() != 'own':
        if selected_model['name'] == 'piper':
            setup_piper()
            update_tts_engines('piper')
        elif selected_model['name'] == 'vits':
            setup_vits()
            update_tts_engines('vits')
        elif selected_model['name'] == 'xtts':
            setup_xtts()
            update_tts_engines('xtts')

        print(f"[{branding}TTS] {selected_model['name']} model downloaded and configuration updated successfully.")
        # Set firstrun_model to false
        set_firstrun_model_false()
        warning_message()
        sys.exit()
else:
    sys.exit()
