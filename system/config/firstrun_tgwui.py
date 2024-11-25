import json
import os
import shutil
import sys
import time
import zipfile
import requests
from tqdm import tqdm
from inputimeout import inputimeout, TimeoutOccurred
from pathlib import Path
from typing import Optional, Dict, List, Any

# Setting the module search path:
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from config import AlltalkTTSEnginesConfig, AlltalkConfig

this_dir = Path(__file__).parent.resolve().parent.resolve().parent.resolve()

def is_running_in_colab() -> bool:
    """Check if the script is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_running_in_docker() -> bool:
    """Check if the script is running in a Docker container."""
    return os.path.exists('/.dockerenv') or (
        os.path.exists('/proc/self/cgroup') and 
        any('docker' in line for line in open('/proc/self/cgroup'))
    )

def download_file(url: str, dest_path: str, timeout: int = 30) -> bool:
    """
    Download a file from a URL with progress bar and error handling.
    
    Args:
        url: URL to download from
        dest_path: Where to save the file
        timeout: Connection timeout in seconds
    
    Returns:
        bool: True if download was successful
    """
    progress_bar = None
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, 
                          desc=f"Downloading {Path(dest_path).name}")
        
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                if data:
                    file.write(data)
                    progress_bar.update(len(data))
        
        if progress_bar.n != total_size and total_size > 0:
            raise requests.exceptions.RequestException(
                f"Incomplete download: {progress_bar.n}/{total_size} bytes"
            )
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        if Path(dest_path).exists():
            Path(dest_path).unlink()
        return False
        
    finally:
        if progress_bar:
            progress_bar.close()

def setup_model(model_name: str) -> bool:
    """
    Set up a specific TTS model.
    
    Args:
        model_name: Name of the model to setup ('piper', 'vits', or 'xtts')
    
    Returns:
        bool: True if setup was successful
    """
    try:
        if model_name == "piper":
            os.makedirs(this_dir / 'models/piper', exist_ok=True)
            os.makedirs(this_dir / 'models/xtts', exist_ok=True)
            return all([
                download_file(
                    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx?download=true",
                    this_dir / "models/piper/en_US-ljspeech-high.onnx"
                ),
                download_file(
                    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json?download=true.json",
                    this_dir / "models/piper/en_US-ljspeech-high.onnx.json"
                )
            ])
            
        elif model_name == "vits":
            os.makedirs(this_dir / 'models/vits', exist_ok=True)
            os.makedirs(this_dir / 'models/xtts', exist_ok=True)
            zip_path = this_dir / 'models/vits/vits.zip'
            
            if not download_file(
                "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--vctk--vits.zip",
                zip_path
            ):
                return False
                
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(this_dir / 'models/vits')
                os.remove(zip_path)
                return True
            except Exception as e:
                print(f"Error extracting VITS model: {str(e)}")
                return False
                
        elif model_name == "xtts":
            os.makedirs(this_dir / 'models/xtts/xttsv2_2.0.3', exist_ok=True)
            base_url = "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/"
            files = [
                "LICENSE.txt", "README.md", "config.json", "model.pth",
                "dvae.pth", "mel_stats.pth", "speakers_xtts.pth", "vocab.json"
            ]
            return all(
                download_file(
                    base_url + file + "?download=true",
                    this_dir / f"models/xtts/xttsv2_2.0.3/{file}"
                ) for file in files
            )
            
        return False
        
    except Exception as e:
        print(f"Error setting up {model_name}: {str(e)}")
        return False

def update_tts_engines(engine: str) -> bool:
    """Update TTS engine configuration with error handling."""
    try:
        AlltalkTTSEnginesConfig.get_instance().change_engine(engine).save()
        return True
    except Exception as e:
        print(f"Error updating TTS engine configuration: {str(e)}")
        return False

def set_firstrun_model_false() -> bool:
    """Set firstrun_model to false with verification."""
    try:
        # Get fresh config instance and update
        config = AlltalkConfig.get_instance()
        config.firstrun_model = False
        config.save()
        
        # Verify the save worked
        config_path = AlltalkConfig.default_config_path()
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            if saved_config.get('firstrun_model', True):
                print(f"Warning: Failed to save firstrun_model=False")
                # Try direct file write as fallback
                saved_config['firstrun_model'] = False
                with open(config_path, 'w') as f:
                    json.dump(saved_config, f, indent=2)
        return True
        
    except Exception as e:
        print(f"Error saving firstrun configuration: {str(e)}")
        return False

def warning_message(branding: str):
    """Display warning messages about installation and compatibility."""
    print(f"[{branding}TTS]")
    print(f"[{branding}TTS] \033[93mIf you have UPGRADED from v1 ensure you have re-installed\033[0m")
    print(f"[{branding}TTS] \033[93mthe requirements. Otherwise you will get failures and errors!\033[0m")
    print(f"[{branding}TTS] \033[93mOn Linux ignore the \033[0m'sparse_attn requires a torch version' \033[93mand\033[0m")
    print(f"[{branding}TTS] \033[0m'using untested triton version' \033[93mmessages.\033[0m")
    print(f"[{branding}TTS]")

def main():
    """Main execution function."""
    try:
        # Load configuration
        with open(AlltalkConfig.default_config_path(), 'r') as file:
            config_json = json.load(file)

        branding = config_json['branding']

        # Handle version 1 config upgrade
        if "ip_address" in config_json:
            print(f"[{branding}TTS] Detected version 1 configuration file.")
            print(f"[{branding}TTS] Upgrading to version 2 configuration file...")

            backup_path = AlltalkConfig.default_config_path() + ".backup"
            shutil.copyfile(AlltalkConfig.default_config_path(), backup_path)

            new_config_template_path = this_dir / 'system/config/confignew.json'
            shutil.copyfile(new_config_template_path, AlltalkConfig.default_config_path())

            print(f"[{branding}TTS] Configuration file upgraded successfully.")

        config = AlltalkConfig.get_instance()
        branding = config.branding

        if not config.firstrun_model:
            return

        # Auto-select Piper for Colab/Docker
        if is_running_in_colab() or is_running_in_docker():
            print(f"[{branding}TTS] Detected Colab/Docker environment - automatically selecting Piper")
            if setup_model('piper'):
                update_tts_engines('piper')
                set_firstrun_model_false()
                warning_message(branding)
            return

        # Normal interactive setup
        models = [
            {"name": "piper", "model": "piper"},
            {"name": "vits", "model": "tts_models--en--vctk--vits"},
            {"name": "xtts", "model": "xttsv2_2.0.3"},
        ]

        print(f"[{branding}TTS]")
        print(f"[{branding}TTS] \033[92mThis is the first time startup. Please download a start TTS model.\033[0m")
        print(f"[{branding}TTS]")
        print(f"[{branding}TTS]    \033[94mAvailable First Time Start-up models:\033[0m")
        print(f"[{branding}TTS]")
        
        for idx, model in enumerate(models):
            print(f"[{branding}TTS]    \033[93m{idx + 1}. \033[94m{model['name']} - {model['model']}\033[0m")
        print(f"[{branding}TTS]    \033[93m{len(models) + 1}.\033[94m I have my own models already\033[0m")
        print(f"[{branding}TTS]")
        print(f"[{branding}TTS]    \033[94mIn \033[91m60 seconds\033[0m \033[94ma Piper model will be \033[91mdownloaded automatically.\033[0m")
        print(f"[{branding}TTS]")

        try:
            user_choice = inputimeout(
                prompt=f"[{branding}TTS]    \033[92mEnter your choice 1-4: \033[0m ",
                timeout=60
            )
        except TimeoutOccurred:
            user_choice = None

        if user_choice is None:
            print(f"[{branding}TTS] No input received. Proceeding with the default model (piper).")
            selected_model = models[0]
        elif user_choice.lower() == 'own' or (user_choice.isdigit() and int(user_choice) == len(models) + 1):
            print(f"[{branding}TTS] If you have your own XTTS models, please move them into the")
            print(f"[{branding}TTS] \033[93m/models/xtts/\033[0m folder")
            
            os.makedirs('models/piper', exist_ok=True)
            os.makedirs('models/vits', exist_ok=True)
            os.makedirs('models/xtts/', exist_ok=True)
            
            update_tts_engines('piper')
            set_firstrun_model_false()
            warning_message(branding)
            return
        elif user_choice.isdigit() and 1 <= int(user_choice) <= len(models):
            selected_model = models[int(user_choice) - 1]
            warning_message(branding)
        else:
            print(f"[{branding}TTS] Invalid choice. Proceeding with the default model (piper).")
            selected_model = models[0]

        if setup_model(selected_model['name']):
            update_tts_engines(selected_model['name'])
            set_firstrun_model_false()
            print(f"[{branding}TTS] {selected_model['name']} model downloaded and configuration updated successfully.")
            warning_message(branding)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
