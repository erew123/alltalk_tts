"""
firstrun.py

This script handles the first-time setup for the Alltalk TTS system. It allows the user to download and configure 
the initial Text-to-Speech (TTS) engine and associated models. It provides automatic detection for Docker or Google 
Colab environments to simplify the setup process, as well as an interactive menu or command-line arguments for manual configuration.

Features:
---------
1. **Automatic Environment Detection**:
    - Detects if the script is running in a Docker container or Google Colab environment.
    - Automatically sets up the Piper TTS engine in these cases.

2. **Command-Line Arguments**:
    - Accepts a `--tts_model` argument to bypass the interactive menu and directly set up a specific TTS engine:
        - `piper`: Sets up the Piper TTS engine.
        - `vits`: Sets up the VITS TTS engine.
        - `xtts`: Sets up the XTTS TTS engine.
        - `none`: Skips model setup entirely.

3. **Interactive Menu**:
    - If no command-line argument is provided and the environment is not Docker/Colab, it presents an interactive 
      menu for the user to select a TTS model.

4. **Configuration Management**:
    - Updates the Alltalk TTS configuration files to reflect the selected TTS engine and marks the initial setup as complete.

Usage:
------
Command-Line:
    python firstrun.py --tts_model [piper|vits|xtts|none]

Interactive Mode:
    python firstrun.py

In Docker/Colab:
    The script automatically selects and sets up the Piper TTS engine without user interaction.

Functions:
----------
- **is_running_in_colab**: Detects if the script is running in Google Colab.
- **is_running_in_docker**: Detects if the script is running in a Docker container.
- **download_file**: Downloads a file with a progress bar and retry logic.
- **setup_piper**: Sets up the Piper TTS engine and downloads associated files.
- **setup_vits**: Sets up the VITS TTS engine and downloads associated files.
- **setup_xtts**: Sets up the XTTS TTS engine and downloads associated files.
- **update_tts_engines**: Updates the configuration to reflect the selected TTS engine.
- **set_firstrun_model_false**: Marks the initial setup as complete in the configuration.
- **warning_message**: Displays post-setup warnings or tips for the user.

Notes:
------
- Ensure network access is available for downloading models during setup.
- Configuration files are automatically updated and backed up if needed.

"""
import argparse
import json
from pathlib import Path
import os
import shutil
import time
import sys
import zipfile
import requests
from tqdm import tqdm
from inputimeout import inputimeout, TimeoutOccurred

# Setting the module search path:
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
from config import AlltalkTTSEnginesConfig, AlltalkConfig #pylint: disable=wrong-import-position

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
        os.path.exists(path) and any('docker' in line for line in open(path, encoding='utf-8'))
    )

def download_file(url: str, dest_path: str, timeout: int = 30, retries: int = 3) -> bool:
    """
    Download a file from a URL to a specified destination path with progress bar.
    
    Args:
        url (str): URL of the file to download.
        dest_path (str): Destination path where the file will be saved.
        timeout (int, optional): Connection and read timeout in seconds. Defaults to 30.
        retries (int, optional): Number of retry attempts for failed downloads. Defaults to 3.
    
    Returns:
        bool: True if the file was successfully downloaded, False otherwise.
    
    Raises:
        ValueError: If the downloaded file size does not match the expected size.
    """
    progress_bar = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempting to download: {url} (Attempt {attempt}/{retries})")
            response = requests.get(url, stream=True, timeout=(timeout, timeout))
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            # Initialize progress bar
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True,
                                desc=f"Downloading {Path(dest_path).name} (Attempt {attempt}/{retries})")
            Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

            with open(dest_path, 'wb') as dwn_file:
                for data in response.iter_content(block_size):
                    if data:
                        dwn_file.write(data)
                        progress_bar.update(len(data))

            if progress_bar:
                progress_bar.close()

            # Verify file size
            if progress_bar.n != total_size and total_size > 0:
                raise ValueError(
                    f"Downloaded size ({progress_bar.n} bytes) does not match expected size ({total_size} bytes)."
                )

            print(f"Download completed: {dest_path}")
            return True

        except requests.Timeout:
            print(f"Timeout error during download: {url}")
        except requests.ConnectionError:
            print(f"Connection error during download: {url}")
        except requests.exceptions.RequestException as e:
            print(f"HTTP error during download: {url}. Details: {str(e)}")
        except ValueError as e:
            print(f"File validation error: {str(e)}")
            if Path(dest_path).exists():
                Path(dest_path).unlink()  # Remove incomplete file
            break  # Stop retrying on file validation errors
        except Exception as e:
            print(f"Unexpected error during download: {str(e)}")
            break  # Stop retrying on unexpected errors
        finally:
            if progress_bar:
                progress_bar.close()

        if Path(dest_path).exists():
            Path(dest_path).unlink()  # Remove incomplete file

        if attempt < retries:
            print(f"Retrying in 5 seconds... (Attempt {attempt + 1}/{retries})")
            time.sleep(5)
        else:
            print(f"Exhausted all retries for {url}.")

    # Return False if all attempts fail
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
        download_file(base_url + each_file + "?download=true", this_dir / f"models/xtts/xttsv2_2.0.3/{each_file}")

def update_tts_engines(engine):
    """Set engine to users choice that matches the download"""
    try:
        config_instance = AlltalkTTSEnginesConfig.get_instance()
        config_instance.change_engine(engine)
        config_instance.save()
    except Exception as e:
        print(f"[{branding}TTS] Error updating TTS engine configuration: {str(e)}")

def set_firstrun_model_false():
    """
    Marks the first-run setup as complete in the Alltalk configuration.

    This function updates the `firstrun_model` flag in the configuration file to `False`,
    indicating that the initial setup has been completed. It attempts to save the updated 
    configuration and verifies the changes, retrying with a direct file write if necessary.
    """
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
    """AllTalk V1 Warning message re updating"""
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
# Argument parser setup
parser = argparse.ArgumentParser(description="TTS Setup Script")
parser.add_argument('--tts_model', type=str, choices=['piper', 'vits', 'xtts', 'none'],
                    help="Specify TTS model to set up (piper, vits, xtts, or none)")
args = parser.parse_args()

# Check if firstrun_model is true
if config.firstrun_model:
    # Handle command-line argument for tts_model
    if args.tts_model:
        if args.tts_model == 'piper':
            setup_piper()
            update_tts_engines('piper')
            set_firstrun_model_false()
        elif args.tts_model == 'vits':
            setup_vits()
            update_tts_engines('vits')
            set_firstrun_model_false()
        elif args.tts_model == 'xtts':
            setup_xtts()
            update_tts_engines('xtts')
            set_firstrun_model_false()
        elif args.tts_model == 'none':
            print(f"[{branding}TTS] No TTS model setup requested.")
        print(f"[{branding}TTS] Setup completed for {args.tts_model}. Exiting.")
        sys.exit()

    # Check for Colab or Docker environment first
    if is_running_in_colab() or is_running_in_docker():
        print(f"[{branding}TTS] Detected Colab/Docker environment - automatically selecting Piper")
        setup_piper()
        update_tts_engines('piper')
        set_firstrun_model_false()
        warning_message()
        sys.exit()

    # Present the menu if no argument is passed and not in Colab/Docker
    print(f"[{branding}TTS]")
    print(f"[{branding}TTS] \033[92mThis is the first-time startup.. Please download a start TTS model. Other TTS engines\033[0m")
    print(f"[{branding}TTS] \033[92mand TTS models can be downloaded/managed in the Gradio Interface `TTS Engines Settings`\033[0m")
    print(f"[{branding}TTS] \033[92mtab after initial setup.\033[0m")
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
        print(f"[{branding}TTS] Please use the Gradio interface TTS Engine Settings > [Engine Name] > Model Download")
        print(f"[{branding}TTS] To download models for your selected TTS Engine. Or use the [Engine Name] help sections")
        print(f"[{branding}TTS] for instructions on using your own TTS models.")
        print(f"[{branding}TTS]")
        os.makedirs('models/piper', exist_ok=True)
        os.makedirs('models/vits', exist_ok=True)
        os.makedirs('models/xtts/', exist_ok=True)
        os.makedirs('models/f5-tts/', exist_ok=True)
        os.makedirs('models/rvc_voices/', exist_ok=True)
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
