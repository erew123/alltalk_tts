"""
Alltalk V2: A Comprehensive TTS and ASR Framework with WebUI Integration

Github: https://github.com/erew123/

This script serves as the core for the Alltalk V2 system, a robust framework combining 
Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) capabilities. It is highly 
configurable and integrates with Text-Generation-WebUI (TGWUI) while also supporting 
standalone deployments. The script initializes configurations, handles subprocess 
management, enables API interactions, and provides transcription utilities.

Key Features:
-------------
1. **TTS Engine Management**:
   - Dynamically load and swap between TTS engines and models.
   - Manage engine-specific configurations and optimize for GPU or CPU usage.

2. **ASR/Transcription Utilities**:
   - Integrates Whisper for audio-to-text transcription.
   - Supports batch processing, live dictation, and various output formats (JSON, TXT, SRT).

3. **WebUI and API Integration**:
   - Extends Text-Generation-WebUI for seamless integration with conversational AI.
   - Provides a Gradio-based interface for managing settings and TTS generation.

4. **Enhanced User Configurations**:
   - Centralized configuration loader for system-wide consistency.
   - Handles first-time setups and runtime environment detection (Docker, Google Colab, etc.).

5. **Audio File Processing**:
   - Supports multiple audio formats with validation, pre-processing (noise reduction, bandpass filtering), 
     and file output management.
   - Automatically organizes transcription files and generates metadata summaries.

6. **Real-Time Feedback and Debugging**:
   - Extensive debug logging for monitoring function entries, file operations, and network requests.
   - Progress tracking for long-running tasks like model loading and transcription.

7. **Command-Line Arguments**:
    - Accepts a `--tts_model` argument to bypass the interactive menu and directly set up a specific TTS engine
      on the first time start-up:
        - `piper`: Sets up the Piper TTS engine.
        - `vits`: Sets up the VITS TTS engine.
        - `xtts`: Sets up the XTTS TTS engine.
        - `none`: Skips model setup entirely.

Requirements:
-------------
- Python 3.8 or later.
- Dependencies listed in the repository's requirements.txt.
- Whisper ASR, PyTorch, Gradio, and other specified modules.

Supported Environments:
-----------------------
- Google Colab
- Docker
- Text-Generation-WebUI (TGWUI)
- Standalone mode
"""
import atexit
import argparse
import gc
import inspect
import importlib
import json
import mimetypes
import os
import platform
import shutil
import signal as system_signal
import subprocess
import sys
import threading
import time
import warnings
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import gradio as gr
import numpy as np
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException
from tqdm import tqdm
import torch
try:
    import whisper
    import soundfile as sf
    import plotly.graph_objects as go
    from scipy import signal as scipy_signal
except ImportError as e:
    print(f"Error: {e}")
    print("=" * 50)
    print("🚨 Missing Dependencies Detected 🚨\n")
    print(
        "It seems you are missing some required packages.\n"
        "To resolve this, please install the dependencies by running:\n"
    )
    print("  🔹 pip install -r system/requirements/requirements_standalone.txt")
    print("  🔹 or use the 'atsetup' command\n")
    print("Once the installation is complete, try running the script again.")
    print("=" * 50)
    sys.exit(1)  # Exit the script to avoid further errors
this_dir = Path(__file__).parent.resolve()

# Note: The following function names are reserved for TGWUI integration.
# When running under text-generation-webui, these functions will be imported from
# system/TGWUI Extension/script.py when in TGWUI's Python environment/extensions dir.
#
# Reserved names:
# - output_modifier
# - input_modifier
# - state_modifier
# - ui
# - history_modifier
# - remove_tts_from_history
# - toggle_text_in_history

TGWUI_AVAILABLE = False

# pylint: disable=import-outside-toplevel
def output_modifier(string, state):
    """Modify chat output (required for TGWUI)"""
    try:
        from .system.TGWUI_Extension.script import (
            output_modifier as tgwui_output_modifier,
        )
    except ImportError:
        from system.TGWUI_Extension.script import (
            output_modifier as tgwui_output_modifier,
        )
    return tgwui_output_modifier(string, state)


def input_modifier(string, state):
    """Modify chat input (required for TGWUI)"""
    try:
        from .system.TGWUI_Extension.script import (
            input_modifier as tgwui_input_modifier,
        )
    except ImportError:
        from system.TGWUI_Extension.script import input_modifier as tgwui_input_modifier
    return tgwui_input_modifier(string, state)


def state_modifier(state):
    """Modify chat state (required for TGWUI)"""
    try:
        from .system.TGWUI_Extension.script import (
            state_modifier as tgwui_state_modifier,
        )
    except ImportError:
        from system.TGWUI_Extension.script import state_modifier as tgwui_state_modifier
    return tgwui_state_modifier(state)


def ui():
    """Create extension UI (required for TGWUI)"""
    try:
        from system.TGWUI_Extension.script import ui as tgwui_ui

        return tgwui_ui()
    except ImportError:
        # Return empty interface if not in TGWUI
        return gr.Blocks()


def history_modifier(history):
    """Modify chat history (required for TGWUI)"""
    try:
        from .system.TGWUI_Extension.script import (
            history_modifier as tgwui_history_modifier,
        )
    except ImportError:
        from system.TGWUI_Extension.script import (
            history_modifier as tgwui_history_modifier,
        )
    return tgwui_history_modifier(history)


def remove_tts_from_history(history):
    """Remove TTS from history (required for TGWUI)"""
    try:
        from .system.TGWUI_Extension.script import (
            remove_tts_from_history as tgwui_remove_tts,
        )
    except ImportError:
        from system.TGWUI_Extension.script import (
            remove_tts_from_history as tgwui_remove_tts,
        )
    return tgwui_remove_tts(history)


def toggle_text_in_history(history):
    """Toggle text in history (required for TGWUI)"""
    try:
        from .system.TGWUI_Extension.script import (
            toggle_text_in_history as tgwui_toggle_text,
        )
    except ImportError:
        from system.TGWUI_Extension.script import (
            toggle_text_in_history as tgwui_toggle_text,
        )
    return tgwui_toggle_text(history)


# pylint: disable=ungrouped-imports,unused-import,import-outside-toplevel

try:
    from modules import chat, shared, ui_chat

    TGWUI_AVAILABLE = True
except ImportError:

    class DummyShared: # pylint: disable=too-few-public-methods
        "fake class relating to how we import or dont import TGWUI's remote extension"
        processing_message = ""

    class DummyState: # pylint: disable=too-few-public-methods
        "fake class relating to how we import or dont import TGWUI's remote extension"
        def __init__(self):
            self.mode = "chat"  # Add default mode

    shared = DummyShared()

# pylint: enable=ungrouped-imports,unused-import,import-outside-toplevel

#########################
# Central config loader #
#########################
# Confguration file management for confignew.json
try:
    from .config import (
        AlltalkConfig,
        AlltalkTTSEnginesConfig,
        AlltalkNewEnginesConfig,
    )  # TGWUI import
    from .system.gradio_pages.help_content import AllTalkHelpContent
    from .system.gradio_pages.alltalk_diskspace import get_disk_interface
    from .system.proxy_module.proxy_manager import ProxyManager
    from .system.proxy_module.interface import create_proxy_interface
except ImportError:
    from config import (
        AlltalkConfig,
        AlltalkTTSEnginesConfig,
        AlltalkNewEnginesConfig,
    )  # Standalone import
    from system.gradio_pages.help_content import AllTalkHelpContent
    from system.gradio_pages.alltalk_diskspace import get_disk_interface
    from system.proxy_module.proxy_manager import ProxyManager
    from system.proxy_module.interface import create_proxy_interface


def initialize_configs():
    """Initialize all configuration instances"""
    config_initalize = AlltalkConfig.get_instance()
    tts_engines_config_initalize = AlltalkTTSEnginesConfig.get_instance()
    new_engines_config_initalize = AlltalkNewEnginesConfig.get_instance()
    return config_initalize, tts_engines_config_initalize, new_engines_config_initalize

# pylint: enable=import-outside-toplevel

# Load in configs
config, tts_engines_config, new_engines_config = initialize_configs()
config.save()  # Force the config file to save in case it was missing new any settings

#########################################
# START-UP # State Dictionary (GLOABLS) #
#########################################
_state = {
    'process': None,
    'running_on_google_colab': False, # Are we running on a Google Colab Server?
    'tunnel_url_1': None, # Used for Google Colab and finding the tunnel URL for API address
    'tunnel_url_2': None, # Used for Google Colab and finding the tunnel URL for Gradio address
    'running_in_docker': False, # Are we running on a Docker Server?
    'docker_url': f"http://localhost:{config.api_def.api_port_number}", # The inital URL used by docker for API communication in gradio
    'alltalk_protocol': "http://", # HTTP is always used, bar Docker or Google Colab. Can be configured here though
    'alltalk_ip_port': f"127.0.0.1:{config.api_def.api_port_number}", # IP/Port used for Docker, Google Colab or default
    'my_current_url': "null",
    'srv_models_available': None, # The current models available for the current TTS engine, pulled from the tts_server's API
    'srv_current_model_loaded': None, # The current model loaded, Initally loaded from central config and later pulled from the tts_server's API
    'srv_engines_available': None, # The current TTS engines available, Initally loaded from central config and later pulled from the tts_server's API
    'srv_current_engine_loaded': None, # The current TTS engines loaded in, Initally loaded from central config and later pulled from the tts_server's API
    'gradio_languages_list': None, # The list of languauges disaplyed in the Gradio interface.
    'whisper_model': None, # Used to track the whisper model used for dictation etc
    'proxy_manager': None # Used for the Proxy manager
}

############################################################
# START-UP # Populate _state Engine info - Gradio Needs it #
############################################################
def initialize_engine_state(_state):
    """Initialize engine state from central config"""
    _state['srv_current_engine_loaded'] = tts_engines_config.engine_loaded
    _state['srv_engines_available'] = tts_engines_config.get_engine_names_available()
    _state['srv_current_model_loaded'] = tts_engines_config.selected_model
    return _state

# Call this after creating _state
_state = initialize_engine_state(_state)

##################################################################
# START-UP # Populate Language Options Choices - Gradio Needs it #
##################################################################
with open(os.path.join(this_dir, "system", "config", "languages.json"), "r", encoding="utf-8") as f:
    _state['gradio_languages_list'] = json.load(f)

##########################
# Central print function #
##########################
# ANSI color codes
BLUE = "\033[94m"
# MAGENTA = "\033[95m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def print_message(message, message_type="standard", component="TTS"):
    """Centralized print function for AllTalk messages
    Args:
        message (str): The message to print
        message_type (str): Type of message (standard/warning/error/debug_*/debug)
        component (str): Component identifier (TTS/ENG/GEN/API/etc.)
    """
    prefix = f"[{config.branding}{component}] "

    if message_type.startswith("debug_"):
        debug_flag = getattr(config.debugging, message_type, False)
        if not debug_flag:
            return

        if message_type == "debug_func" and "Function entry:" in message:
            message_parts = message.split("Function entry:", 1)
            print(
                f"{prefix}{BLUE}Debug{RESET} {YELLOW}{message_type}{RESET} Function entry:{GREEN}{message_parts[1]}{RESET} script.py"
            )
        else:
            print(f"{prefix}{BLUE}Debug{RESET} {YELLOW}{message_type}{RESET} {message}")

    elif message_type == "debug":
        print(f"{prefix}{BLUE}Debug{RESET} {message}")

    elif message_type == "warning":
        print(f"{prefix}{YELLOW}Warning{RESET} {message}")

    elif message_type == "error":
        print(f"{prefix}{RED}Error{RESET} {message}")

    else:
        print(f"{prefix}{message}")


def debug_func_entry():
    """Print debug message for function entry if debug_func is enabled"""
    if config.debugging.debug_func:
        current_func = inspect.currentframe().f_back.f_code.co_name
        print_message(f"Function entry: {current_func}", "debug_func")


###########################
# Central config updaters #
###########################
def update_settings_at(
    delete_output_wavs,
    gradio_interface,
    gradio_port_number,
    upd_output_folder,
    api_port_number,
    gr_debug_tts,
    transcode_audio_format,
    generate_help_page,
    voice2rvc_page,
    tts_generator_page,
    tts_engines_settings_page,
    alltalk_documentation_page,
    api_documentation_page,
):
    """Update AllTalk main settings using the centralized config system"""
    debug_func_entry()
    try:
        # Get the current config instance
        upd_set_config = AlltalkConfig.get_instance()

        # Update main settings
        upd_set_config.delete_output_wavs = delete_output_wavs
        upd_set_config.gradio_interface = gradio_interface == "Enabled"
        upd_set_config.output_folder = upd_output_folder
        upd_set_config.api_def.api_port_number = api_port_number
        upd_set_config.gradio_port_number = gradio_port_number
        upd_set_config.transcode_audio_format = transcode_audio_format

        # Update debugging options
        for key in vars(upd_set_config.debugging):
            if not key.startswith("_"):  # Skip private attributes
                setattr(upd_set_config.debugging, key, key in gr_debug_tts)

        # Update gradio pages settings
        upd_set_config.gradio_pages.Generate_Help_page = generate_help_page
        upd_set_config.gradio_pages.Voice2RVC_page = voice2rvc_page
        upd_set_config.gradio_pages.TTS_Generator_page = tts_generator_page
        upd_set_config.gradio_pages.TTS_Engines_Settings_page = tts_engines_settings_page
        upd_set_config.gradio_pages.alltalk_documentation_page = alltalk_documentation_page
        upd_set_config.gradio_pages.api_documentation_page = api_documentation_page

        # Save the updated configuration
        upd_set_config.save()
        # Tell tts_server.py to update
        get_alltalk_settings()

        print_message("Default Settings Saved")
        return "Settings updated successfully!"
    except (AttributeError, TypeError) as e:
        print_message(
            f"Configuration structure error: {str(e)}", 
            message_type="error"
        )
        return "Error updating settings: Invalid configuration structure"
    except (OSError, IOError) as e:
        print_message(
            f"File system error while saving configuration: {str(e)}", 
            message_type="error"
        )
        return "Error saving settings: File system error"
    except ValueError as e:
        print_message(
            f"Invalid value provided for configuration: {str(e)}", 
            message_type="error"
        )
        return "Error updating settings: Invalid value provided"


def update_settings_api(
    api_length_stripping,
    api_legacy_ip_address,
    api_allowed_filter,
    api_max_characters,
    api_use_legacy_api,
    api_text_filtering,
    api_narrator_enabled,
    api_text_not_inside,
    api_language,
    api_output_file_name,
    api_output_file_timestamp,
    api_autoplay,
    api_autoplay_volume,
):
    """Update API settings using the centralized config system"""
    debug_func_entry()
    try:
        # Get the current config instance
        upd_config = AlltalkConfig.get_instance()

        # Update API settings
        upd_config.api_def.api_length_stripping = api_length_stripping
        upd_config.api_def.api_legacy_ip_address = api_legacy_ip_address
        upd_config.api_def.api_allowed_filter = api_allowed_filter
        upd_config.api_def.api_max_characters = api_max_characters
        upd_config.api_def.api_use_legacy_api = (
            api_use_legacy_api == "AllTalk v1 API (Legacy)"
        )
        upd_config.api_def.api_text_filtering = api_text_filtering
        upd_config.api_def.api_narrator_enabled = api_narrator_enabled
        upd_config.api_def.api_text_not_inside = api_text_not_inside
        upd_config.api_def.api_language = api_language
        upd_config.api_def.api_output_file_name = api_output_file_name
        upd_config.api_def.api_output_file_timestamp = (
            api_output_file_timestamp == "Timestamp files"
        )
        upd_config.api_def.api_autoplay = api_autoplay == "Play remotely"
        upd_config.api_def.api_autoplay_volume = api_autoplay_volume

        # Save the updated configuration
        upd_config.save()
        # Tell tts_server.py to update
        get_alltalk_settings()

        print_message("API Settings Saved")
        return "Default API settings updated successfully!"
    except (AttributeError, TypeError) as e:
        print_message(
            f"Configuration structure error: {str(e)}", 
            message_type="error"
        )
        return "Error updating settings: Invalid configuration structure"
    except (OSError, IOError) as e:
        print_message(
            f"File system error while saving configuration: {str(e)}", 
            message_type="error"
        )
        return "Error saving settings: File system error"
    except ValueError as e:
        print_message(
            f"Invalid value provided for configuration: {str(e)}", 
            message_type="error"
        )
        return "Error updating settings: Invalid value provided"


def update_settings_tgwui(activate, autoplay, show_text, language, narrator_enabled):
    """Update Text-gen-webui settings using the centralized config system"""
    debug_func_entry()
    try:
        # Get the current config instance
        tgwui_config = AlltalkConfig.get_instance()

        # Update TGWUI settings
        tgwui_config.tgwui.tgwui_activate_tts = activate == "Enabled"
        tgwui_config.tgwui.tgwui_autoplay_tts = autoplay == "Enabled"
        tgwui_config.tgwui.tgwui_show_text = show_text == "Enabled"
        tgwui_config.tgwui.tgwui_language = language
        tgwui_config.tgwui.tgwui_narrator_enabled = narrator_enabled

        # Save the updated configuration
        tgwui_config.save()

        print_message("Default Text-gen-webui Settings Saved")
        return "Settings updated successfully!"
    except (AttributeError, TypeError) as e:
        error_msg = f"Configuration structure error: {str(e)}"
        return_message = "Error updating settings: Invalid configuration structure"
    except (OSError, IOError) as e:
        error_msg = f"File system error while saving configuration: {str(e)}"
        return_message = "Error saving settings: File system error"
    except ValueError as e:
        error_msg = f"Invalid value provided for configuration: {str(e)}"
        return_message = "Error updating settings: Invalid value provided"

    print_message(error_msg, message_type="error")
    return return_message


def update_rvc_settings(
    rvc_enabled,
    rvc_char_model_file,
    rvc_narr_model_file,
    split_audio,
    autotune,
    pitch,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0method,
    embedder_model,
    training_data_size,
    progress=None,
):
    """Update RVC settings using the centralized config system"""
    debug_func_entry()

    try:
        # Get the current config instance
        rvc_set_config = AlltalkConfig.get_instance()

        # Update RVC settings
        rvc_set_config.rvc_settings.rvc_enabled = rvc_enabled
        rvc_set_config.rvc_settings.rvc_char_model_file = rvc_char_model_file
        rvc_set_config.rvc_settings.rvc_narr_model_file = rvc_narr_model_file
        rvc_set_config.rvc_settings.split_audio = split_audio
        rvc_set_config.rvc_settings.autotune = autotune
        rvc_set_config.rvc_settings.pitch = pitch
        rvc_set_config.rvc_settings.filter_radius = filter_radius
        rvc_set_config.rvc_settings.index_rate = index_rate
        rvc_set_config.rvc_settings.rms_mix_rate = rms_mix_rate
        rvc_set_config.rvc_settings.protect = protect
        rvc_set_config.rvc_settings.hop_length = hop_length
        rvc_set_config.rvc_settings.f0method = f0method
        rvc_set_config.rvc_settings.embedder_model = embedder_model
        rvc_set_config.rvc_settings.training_data_size = training_data_size

        def ensure_directory_exists(directory):
            """Confirm a directory exists"""
            if not os.path.exists(directory):
                os.makedirs(directory)

        def load_file_urls(json_path):
            """Load and return JSON data from specified file path."""
            with open(json_path, "r", encoding="utf-8") as json_file:
                return json.load(json_file)

        def download_file(useurl, dest_path):
            """
            Download file from URL and save to specified path.

            Args:
                url (str): URL to download from
                dest_path (str): Path where file will be saved
            """
            # First number (5) = connection timeout (time to establish connection)
            # Second number (30) = read timeout (time between bytes received)
            file_response = requests.get(useurl, stream=True, timeout=(5, 30))
            file_response.raise_for_status()
            with open(dest_path, "wb") as downloaded_file:
                for chunk in file_response.iter_content(chunk_size=8192):
                    downloaded_file.write(chunk)

        if rvc_enabled:
            base_dir = os.path.join(this_dir, "models", "rvc_base")
            rvc_voices_dir = os.path.join(this_dir, "models", "rvc_voices")
            ensure_directory_exists(base_dir)
            ensure_directory_exists(rvc_voices_dir)
            json_path = os.path.join(
                this_dir, "system", "tts_engines", "rvc_files.json"
            )
            file_urls = load_file_urls(json_path)
            for idx, file in enumerate(file_urls):
                if not os.path.exists(os.path.join(base_dir, file)):
                    progress(
                        idx / len(file_urls),
                        desc=f"Downloading Required RVC Files: {file}...",
                    )
                    print(
                        f"[{config.branding}TTS] Downloading {file}..."
                    )  # Print statement for terminal
                    download_file(file_urls[file], os.path.join(base_dir, file))
            download_result = (
                "All RVC Base Files are present."
                if len(file_urls) > 0
                else "All files are present."
            )
            print_message(download_result)

        # Save the updated configuration
        rvc_set_config.save()
        # Tell tts_server.py to update
        get_alltalk_settings()

        return "RVC settings updated successfully!"
    except RequestException as e:
        error_msg = f"Error downloading RVC files: {str(e)}"
        return_message = "Error downloading RVC files: Network or server error"
    except json.JSONDecodeError as e:
        error_msg = f"Error reading RVC configuration file: {str(e)}"
        return_message = "Error with RVC configuration: Invalid JSON format"
    except (AttributeError, TypeError) as e:
        error_msg = f"Configuration structure error: {str(e)}"
        return_message = "Error updating RVC settings: Invalid configuration structure"
    except (OSError, IOError) as e:
        error_msg = f"File system error while saving/creating directories: {str(e)}"
        return_message = "Error with RVC files: File system error"
    print_message(error_msg, message_type="error")
    return return_message

def update_proxy_settings(
    proxy_enabled,
    start_on_startup,
    gradio_proxy_enabled,
    api_proxy_enabled,
    external_ip,
    gradio_cert_name,
    api_cert_name,
    cert_validation,
    logging_enabled,
    log_level
):
    """Update proxy settings using the centralized config system"""
    debug_func_entry()
    try:
        # Get the current config instance
        config = AlltalkConfig.get_instance()

        # Update proxy settings
        config.proxy_settings.proxy_enabled = proxy_enabled
        config.proxy_settings.start_on_startup = start_on_startup
        config.proxy_settings.gradio_proxy_enabled = gradio_proxy_enabled
        config.proxy_settings.api_proxy_enabled = api_proxy_enabled
        config.proxy_settings.external_ip = external_ip
        config.proxy_settings.gradio_cert_name = gradio_cert_name
        config.proxy_settings.api_cert_name = api_cert_name
        config.proxy_settings.cert_validation = cert_validation
        config.proxy_settings.logging_enabled = logging_enabled
        config.proxy_settings.log_level = log_level

        # Save the updated configuration
        config.save()

        print_message("Proxy Settings Saved")
        return "Proxy settings updated successfully!"
    except Exception as e:
        print_message(f"Error updating proxy settings: {str(e)}", message_type="error")
        return f"Error updating proxy settings: {str(e)}"

# Add to your state dictionary initialization
def initialize_proxy_state(_state):
    """Initialize proxy state"""
    config = AlltalkConfig.get_instance()
    _state['proxy_process'] = None
    _state['proxy_status'] = 'stopped'
    _state['proxy_enabled'] = config.proxy_settings.proxy_enabled
    _state['proxy_gradio_enabled'] = config.proxy_settings.gradio_proxy_enabled
    _state['proxy_api_enabled'] = config.proxy_settings.api_proxy_enabled
    return _state

# Modify your existing initialize_engine_state function:
def initialize_engine_state(_state):
    """Initialize engine state from central config"""
    _state = initialize_proxy_state(_state)  # Add proxy initialization
    _state['srv_current_engine_loaded'] = tts_engines_config.engine_loaded
    _state['srv_engines_available'] = tts_engines_config.get_engine_names_available()
    _state['srv_current_model_loaded'] = tts_engines_config.selected_model
    return _state

###########################################################################
# START-UP # Silence Character Normaliser when it checks the Ready Status #
###########################################################################
warnings.filterwarnings(
    "ignore", message="Trying to detect encoding from a tiny portion"
)

###########################################
# START-UP # AllTalk allowed startup time #
###########################################
startup_wait_time = 240

# You can change the above setting to a larger number to allow AllTAlk more time to start up.
# The default setting is 240 seconds (4 minutes). If its taking longer though, you may have a
# Very old system or system issue.

##############################################
# START-UP # Load confignew.json into params #
##############################################
config = AlltalkConfig.get_instance()
github_site = "erew123"
github_repository = "alltalk_tts"
github_branch = "alltalkbeta"
current_folder = os.path.basename(os.getcwd())
output_folder = config.get_output_directory()
delete_output_wavs_setting = config.delete_output_wavs
gradio_enabled = config.gradio_interface
script_path = this_dir / "tts_server.py"


############################################
# START-UP # Display initial splash screen #
############################################
# pylint: disable=line-too-long,anomalous-backslash-in-string
print_message(
    "\033[94m    _    _ _ \033[1;35m_____     _ _     \033[0m  _____ _____ ____  "
)
print_message(
    "\033[94m   / \  | | |\033[1;35m_   _|_ _| | | __ \033[0m |_   _|_   _/ ___| "
)
print_message(
    "\033[94m  / _ \ | | |\033[1;35m | |/ _` | | |/ / \033[0m   | |   | | \___ \ "
)
print_message(
    "\033[94m / ___ \| | |\033[1;35m | | (_| | |   <  \033[0m   | |   | |  ___) |"
)
print_message(
    "\033[94m/_/   \_\_|_|\033[1;35m |_|\__,_|_|_|\_\ \033[0m   |_|   |_| |____/ "
)
print_message("")
# pylint: enable=line-too-long,anomalous-backslash-in-string


#############################################################################
# START-UP # Check current folder name has dashes '-' in it and error if so #
#############################################################################
# Get the directory of the current script
this_script_path = Path(__file__).resolve()
this_script_dir = this_script_path.parent
# Get the current folder name
this_current_folder = this_script_dir.name
if "-" in this_current_folder:
    print_message("")
    print_message( # pylint: disable=line-too-long
        "The current folder name contains a dash ('\033[93m-\033[0m') and this causes errors/issues. Please ensure",
        message_type="warning",
    )
    # pylint: disable=line-too-long
    print_message(
        "the folder name does not have a dash e.g. rename ('\033[93malltalk_tts-main\033[0m') to ('\033[93malltalk_tts\033[0m').",
        message_type="warning",
    )
    print_message("")
    print_message(
        "\033[92mCurrent folder:\033[0m {this_current_folder}", message_type="warning"
    )
    sys.exit(1)

# pylint: disable=ungrouped-imports,unused-import,import-outside-toplevel,import-error

##############################################
# START-UP # Check if we are on Google Colab #
##############################################
def check_google_colab():
    """
    Test if we are running on a google colab server
    """
    debug_func_entry()
    try:
        import google.colab
        return True
    except ImportError:
        return False
    
_state['running_on_google_colab'] = check_google_colab()

###############################################################################
# START-UP # Test if we are running within Text-gen-webui or as a Standalone  #
###############################################################################
try:
    from modules import chat, shared, ui_chat
    from modules.logging_colors import logger
    from modules.ui import create_refresh_button
    from modules.utils import gradio

    print_message("\033[92mStart-up Mode     : \033[93mText-gen-webui mode\033[0m")
    running_in_standalone = False
    running_in_tgwui = True
except ModuleNotFoundError:
    running_in_standalone = True
    running_in_tgwui = False
    print_message("\033[92mStart-up Mode     : \033[93mStandalone mode\033[0m")

# pylint: enable=ungrouped-imports,unused-import,import-outside-toplevel,import-error

######################################################
# START-UP # Check if this is a first time start-up  #
######################################################
def run_firsttime_script(tts_model=None):
    """
    Run the first time startup script based on the current environment
    (Google Colab, standalone, or TGWUI). Optionally, pass a TTS model
    argument to the script for direct configuration.
    
    Args:
        tts_model (str): Optional. TTS model to set up ('piper', 'vits', 'xtts', or 'none').
    """
    debug_func_entry()
    try:
        # Get the current config instance
        firstrun_config = AlltalkConfig.get_instance()
        # Determine the script path based on environment
        if _state['running_on_google_colab']:
            firstrun_script_path = "/content/alltalk_tts/system/config/firstrun.py"
        elif running_in_standalone:
            firstrun_script_path = os.path.join(this_dir, "system", "config", "firstrun.py")
        elif running_in_tgwui:
            firstrun_script_path = os.path.join(
                this_dir, "system", "config", "firstrun_tgwui.py"
            )
        else:
            firstrun_script_path = os.path.join(this_dir, "system", "config", "firstrun.py")
        # Prepare the subprocess command
        command = [sys.executable, firstrun_script_path]
        # Append the --tts_model argument if provided
        if tts_model:
            if tts_model not in ['piper', 'vits', 'xtts', 'none']:
                raise ValueError(f"Invalid tts_model value: {tts_model}")
            command.extend(['--tts_model', tts_model])
        # Run the script
        subprocess.run(command, check=True)
        # Reload the configuration after script execution
        firstrun_config.reload()
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running first-time setup script: {str(e)}"
        print_message(error_msg, message_type="error")
    except (FileNotFoundError, PermissionError) as e:
        error_msg = f"File system error during first-time setup: {str(e)}"
        print_message(error_msg, message_type="error")
    except ValueError as e:
        error_msg = f"Invalid argument passed to first-time setup script: {str(e)}"
        print_message(error_msg, message_type="error")

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Run the first-time setup script.")
parser.add_argument(
    "--tts_model",
    type=str,
    choices=["piper", "vits", "xtts", "none"],
    help="Specify the TTS model to set up (piper, vits, xtts, or none).",
)
# Parse the arguments
args = parser.parse_args()
# Call the function to run the startup script
run_firsttime_script(tts_model=args.tts_model)


###########################################################
# START-UP # Delete files in outputs folder if configured #
###########################################################
def delete_old_files(folder_path, amt_days_to_keep):
    """
    Delete files in the output folder that are X days old
    """
    debug_func_entry()
    current_time = datetime.now()
    print_message(
        "\033[92mWAV file deletion    :\033[93m", delete_output_wavs_setting, "\033[0m"
    )
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            age = current_time - file_creation_time
            if age > timedelta(days=amt_days_to_keep):
                os.remove(file_path)


# Check and perform file deletion
if delete_output_wavs_setting.strip().lower() == "disabled":
    print_message("\033[92mWAV file deletion :\033[93m Disabled\033[0m")
else:
    try:
        days_to_keep = int(delete_output_wavs_setting.split()[0])
        delete_old_files(output_folder, days_to_keep)
    except ValueError:
        # pylint: disable=line-too-long
        print_message(
            "\033[92mWAV file deletion :\033[93m Invalid setting for deleting old wav files. Please use 'Disabled' or 'X Days' format\033[0m"
        )


#####################################################################
# START-UP # Check Githubs last update and output into splashscreen #
#####################################################################
def format_datetime(iso_str):
    """
    Formats an ISO datetime string into a human-readable format with ordinal day numbers.
    Example: Converts "2024-03-19T10:30:00Z" to "19th March 2024 at 10:30"
    """
    debug_func_entry()

    def _ordinal(n):
        """Helper function to convert numbers to ordinal form (1st, 2nd, 3rd, etc)"""
        debug_func_entry()
        suffix = "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime(f"{_ordinal(dt.day)} %B %Y at %H:%M")


def fetch_latest_commit_sha_and_date(owner, repo, branch):
    """
    Fetch the latest commit SHA and date from a GitHub repository branch.
    
    Args:
        owner (str): GitHub repository owner
        repo (str): Repository name 
        branch (str): Branch name to check
        
    Returns:
        tuple: (commit_sha, commit_date) or (None, None) if fetch fails
    """
    debug_func_entry()
    # Modified URL to include the branch
    github_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
    try:
        # Add timeout for both connect and read operations
        github_response = requests.get(github_url, timeout=(5, 30))
        if github_response.status_code == 200:
            commit_data = github_response.json()
            github_latest_commit_sha = commit_data["sha"]
            github_latest_commit_date = commit_data["commit"]["committer"]["date"]
            return github_latest_commit_sha, github_latest_commit_date
        # pylint: disable=line-too-long
        print_message(
            f"\033[92mGithub updated    :\033[91m Failed to fetch the latest commit from branch {branch} due to an unexpected response from GitHub"
        )
        return None, None
    except (RequestsConnectionError, requests.Timeout):  # Added Timeout to caught exceptions
        print_message(
            "\033[92mGithub updated    :\033[91m Could not reach GitHub to check for updates\033[0m"
        )
        return None, None


def read_or_initialize_sha(file_path, owner, repo, branch):
    """
    Read SHA from existing file or initialize with latest commit SHA if file doesn't exist.
    """
    debug_func_entry()
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            current_data = json.load(file)
            return current_data.get("last_known_commit_sha")
    else:
        # File doesn't exist, fetch the latest SHA and create the file
        current_commit_sha, _ = fetch_latest_commit_sha_and_date(owner, repo, branch)
        if current_commit_sha:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump({"last_known_commit_sha": current_commit_sha}, file)
            return current_commit_sha
        return None


def update_sha_file(file_path, new_sha):
    """
    Update the stored commit SHA in the specified file.
    """
    debug_func_entry()
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump({"last_known_commit_sha": new_sha}, file)


# Define the file path based on your directory structure
sha_file_path = this_dir / "system" / "config" / "at_github_sha.json"

# Read or initialize the SHA
last_known_commit_sha = read_or_initialize_sha(
    sha_file_path, github_site, github_repository, github_branch
)

# Fetch the latest commit SHA and date from the specific branch
latest_commit_sha, latest_commit_date = fetch_latest_commit_sha_and_date(
    github_site, github_repository, github_branch
)

formatted_date = (
    format_datetime(latest_commit_date) if latest_commit_date else "an unknown date"
)

if latest_commit_sha and latest_commit_sha != last_known_commit_sha:
    print_message(
        f"\033[92mGithub updated    :\033[93m {formatted_date} \033[92mBranch:\033[93m {github_branch}\033[0m"
    )
    # Update the file with the new SHA
    update_sha_file(sha_file_path, latest_commit_sha)
elif latest_commit_sha == last_known_commit_sha:
    print_message(
        f"\033[92mGithub updated    :\033[93m {formatted_date} \033[92mBranch:\033[93m {github_branch}\033[0m"
    )


##################################################
# START-UP # Configure the subprocess handler ####
##################################################
def signal_handler(sig, frame): # pylint: disable=unused-argument
    """Handle Ctrl+C signal by saving config, terminating subprocess, and exiting gracefully."""
    debug_func_entry()
    config.save()
    print_message(
        "\033[94mReceived Ctrl+C, terminating subprocess. Kill your Python processes if this fails to exit.\033[92m"
    )
    if _state["process"].poll() is None:
        _state["process"].terminate()
        _state["process"].wait()  # Wait for the subprocess to finish
    sys.exit(0)

#####################################################################
# START-UP # Start the Subprocess and Check for Google Colab/Docker #
#####################################################################
# pylint: disable=ungrouped-imports,unused-import,import-outside-toplevel,import-error
if check_google_colab():
    try:
        with open("/content/alltalk_tts/googlecolab.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            _state['tunnel_url_1'], _state['tunnel_url_2'] = data.get("google_ip_address", [None, None])
    except FileNotFoundError:
        print_message("Could not find IP address")
        _state['tunnel_url_1'], _state['tunnel_url_2'] = None, None
else:
    _state['running_on_google_colab'] = False
# pylint: enable=ungrouped-imports,unused-import,import-outside-toplevel


# Attach the signal handler to the SIGINT signal (Ctrl+C)
system_signal.signal(system_signal.SIGINT, signal_handler)

# Check if we're running in docker
if os.path.isfile("/.dockerenv") and "google.colab" not in sys.modules:
    print_message("")
    print_message("\033[94mRunning in Docker environment. Please note:\033[0m")
    print_message("Docker environments have various considerations:")
    print_message(
        "AllTalk v2 is running 2x web servers. One webserver is the API address that exernal TTS"
    )
    print_message(
        "generation requests are sent to. The other webserver is the Gradio management interface."
    )
    print_message(
        "If you want to access and use both API calls and the Gradio interface, in a docker style"
    )
    print_message(
        "environment, you will need to somehow make these accessable, depending on your scenario:"
    )
    print_message(" 1. A Local Area Network (LAN) scenario:")
    print_message("    - Ensure you've exposed the Gradio and API ports correctly.")
    print_message("    - The application should work as expected.")
    print_message("2. Internet/Remotely accessed scenario:")
    print_message(
        "    - You'll need to set up a secure tunnel or VPN to the host server."
    )
    print_message(
        "    - If you want just API requests, you can map just the API and control AllTalk via API calls."
    )
    print_message(
        "    - If you want both API and Gradio interfaces you need to make both accessible."
    )
    print_message(" 3. Default Addresses:")
    print_message(
        f"    - Internal API address: http://localhost:{config.api_def.api_port_number}"
    )
    print_message(
        f"    - Internal Gradio address: http://localhost:{config.gradio_port_number}"
    )
    print_message(" Tunnel Setup:")
    print_message(
        "    - If using a tunnel, you'll need to provide the API external URL/Address in Gradio."
    )
    print_message(
        "    - Look for 'API URL/Address' option in the Gradio interface `Generate` page."
    )
    _state['running_in_docker'] = True
else:
    _state['running_in_docker'] = False

# Start the subprocess (now unified for both Docker and non-Docker environments)
# Not using 'with' as we want the process to run in the background
# pylint: disable=consider-using-with
_state["process"] = subprocess.Popen([sys.executable, script_path])

# Check if the subprocess has started successfully
if _state["process"].poll() is None:
    if _state['running_on_google_colab']:
        print_message("")
        print_message("\033[94mGoogle Colab Detected\033[00m")
        print_message("")
else:
    print_message(
        "TTS Subprocess Webserver failing to start process", message_type="warning"
    )
    print_message(
        f"It could be that you have something on port: {config.port_number}",
        message_type="warning",
    )
    print_message(
        "Or you have not started in a Python environment with all the necessary bits installed",
        message_type="warning",
    )
    # pylint: disable=line-too-long
    print_message(
        "Check you are starting Text-generation-webui with either the start_xxxxx file or the Python environment with cmd_xxxxx file.",
        message_type="warning",
    )
    # pylint: disable=line-too-long
    print_message(
        "xxxxx is the type of OS you are on e.g. windows, linux or mac.",
        message_type="warning",
    )
    # pylint: disable=line-too-long
    print_message(
        "Alternatively, you could check no other Python processes are running that shouldn't be e.g. Restart your computer is the simple way.",
        message_type="warning",
    )
    # Cleanly kill off this script, but allow text-generation-webui to keep running, albeit without this alltalk_tts
    sys.exit(1)

timeout = startup_wait_time  # Gather timeout setting from startup_wait_time
initial_delay = 5  # Initial delay before starting the check loop
warning_delay = 60  # Delay before displaying warnings

# Introduce a delay before starting the check loop
time.sleep(initial_delay)

start_time = time.time()
warning_displayed = False

url = f"http://localhost:{config.api_def.api_port_number}/api/ready"
while time.time() - start_time < timeout:
    try:
        response = requests.get(url, timeout=(5, 30))
        if response.status_code == 200 and response.text == "Ready":
            break
    except RequestException as e:
        # Log the exception if needed
        pass

    if not warning_displayed and time.time() - start_time >= warning_delay:
        print_message(
            "TTS Engine has NOT started up yet. Will keep trying for "
            + str(timeout)
            + " seconds maximum. Please wait.",
            message_type="warning",
        )
        print_message(
            "Mechanical hard drives and a slow PCI BUS are examples of things that can affect load times.",
            message_type="warning",
        )
        print_message(
            "Some TTS engines index their AI TTS models on loading, which can be slow on CPU or old systems.",
            message_type="warning",
        )
        print_message(
            "Using one of the other TTS engines on slower systems can help ease this issue.",
            message_type="warning",
        )
        warning_displayed = True

    time.sleep(1)
else:
    print_message("")
    print_message(
        "Startup timed out. Full help available here https://github.com/erew123/alltalk_tts#-help-with-problems"
    )
    print_message(
        "On older systems, you may wish to open and edit script.py with a text editor and change the"
    )
    print_message(
        "startup_wait_time = 240 setting to something like startup_wait_time = 460 as this will allow"
    )
    print_message(
        "AllTalk more time (6 mins) to try load the model into your VRAM. Otherwise, please visit the GitHub for"
    )
    print_message("a list of other possible troubleshooting options.")
    # Cleanly kill off this script, but allow text-generation-webui to keep running, albeit without this alltalk_tts
    sys.exit(1)

if _state['running_on_google_colab']:
    print_message("")
    print_message("\033[94mAPI Address :\033[00m \033[92m" + _state['tunnel_url_1'] + "\033[00m")
    print_message("\033[94mGradio Light:\033[00m \033[92m" + _state['tunnel_url_2'] + "\033[00m")
    print_message(
        "\033[94mGradio Dark :\033[00m \033[92m"
        + _state['tunnel_url_2']
        + "?__theme=dark\033[00m"
    )
    print_message("")
else:
    print_message("")
    print_message(
        "\033[94mAPI Address :\033[00m \033[92m127.0.0.1:"
        + str(config.api_def.api_port_number)
        + "\033[00m"
    )
    if config.launch_gradio:
        print_message(
            "\033[94mGradio Light:\033[00m \033[92mhttp://127.0.0.1:"
            + str(config.gradio_port_number)
            + "\033[00m"
        )
        print_message(
            "\033[94mGradio Dark :\033[00m \033[92mhttp://127.0.0.1:"
            + str(config.gradio_port_number)
            + "?__theme=dark\033[00m"
        )
    print_message("")

print_message(
    "\033[94mAllTalk WIKI:\033[00m \033[92mhttps://github.com/erew123/alltalk_tts/wiki\033[00m"
)
print_message(
    "\033[94mErrors Help :\033[00m \033[92mhttps://github.com/erew123/alltalk_tts/wiki/Error-Messages-List\033[00m"
)
print_message("")


#########################################
# START-UP # Espeak-ng check on Windows #
#########################################
def check_espeak_ng():
    """
    Test to see if espeak-ng is installed and available. It's required
    by various TTS engines.
    """
    debug_func_entry()

    def print_espeak_warning(platform_name, install_instructions):
        print_message("")
        print_message(
            f"Espeak-ng for {platform_name}\033[91m WAS NOT FOUND. \033[0mYou can install\033[0m",
            message_type="warning",
        )
        print_message(
            f"\033[0mit using {install_instructions}\033[0m",
            message_type="warning",
        )

    try:
        subprocess.run(
            ["espeak-ng", "--version"], capture_output=True, text=True, check=True
        )
        return
    except FileNotFoundError:
        if platform.system() == "Windows":
            print_espeak_warning(
                "Windows", r"this location \033[93m\alltalk_tts\system\espeak-ng\\"
            )
            print_message(
                "Then close this command prompt window and open a new",
                message_type="warning",
            )
            print_message("command prompt, before re-starting.", message_type="warning")
        elif platform.system() == "Darwin":  # macOS
            print_espeak_warning(
                "macOS", "\033[93mHomebrew: brew install espeak-ng\033[0m"
            )
        else:  # Linux
            print_espeak_warning(
                "Linux",
                "\033[93myour package manager, e.g., apt-get install espeak-ng\033[0m",
            )
    except subprocess.CalledProcessError:
        # Handle cases where `espeak-ng` exists but fails to run
        print_message("Error running espeak-ng. Please check the installation.", message_type="error")

    print_message("")



check_espeak_ng()

####################################
# START-UP # Subprecess management #
####################################


def start_subprocess():
    """Start a new subprocess if none is running. Return status message."""
    debug_func_entry()
    if _state["process"] is None or _state["process"].poll() is not None:
        # pylint: disable=consider-using-with  # Process needs to run in background
        _state["process"] = subprocess.Popen([sys.executable, script_path])
        return "Subprocess started."
    return "Subprocess is already running."


def stop_subprocess():
    """Terminate running subprocess and wait for completion. Return status message."""
    debug_func_entry()
    if _state["process"] is not None:
        _state["process"].terminate()
        _state["process"].wait()
        _state["process"] = None
        return "Subprocess stopped."
    return "Subprocess is not running."


def restart_subprocess():
    """Stop current subprocess, display TTS engine swap message, and start new subprocess."""
    debug_func_entry()
    stop_subprocess()
    print_message("")
    print_message("\033[94mSwapping TTS Engine. Please wait.\033[00m")
    print_message("")
    return start_subprocess()


def check_subprocess_status():
    """Check if subprocess is currently running and return status message."""
    debug_func_entry()
    if _state["process"] is None or _state["process"].poll() is not None:
        return "Subprocess is not running."
    return "Subprocess is running."


###################################################################
# START-UP # Register the termination code to be executed at exit #
###################################################################
atexit.register(lambda: _state["process"].terminate() if _state["process"].poll() is None else None)

##########################
# Setup global variables #
##########################
# deepspeed_installed = True
tgwui_lovram = False
tgwui_deepspeed = False
# Create a global lock for tracking TTS generation occuring
process_lock = threading.Lock()
# Pull the connection timeout value for communication requests with the AllTalk remote server
connection_timeout = 10
# Used to detect if a model is loaded in to AllTalk server to block TTS genereation if needed.
tts_model_loaded = None

#########################
# Endpoint API Builders #
#########################
def build_url(endpoint, include_api=True):
    """
    Build URL for standard API endpoints using static predefined URLs.
    Used for endpoints that don't need dynamic domain handling.

    Examples:
        - voice2rvc
        - reload
        - stop-generation

    Args:
        endpoint (str): The API endpoint to append to the base URL
        include_api (bool): Whether to include /api/ in the URL
            True: Uses '/api/'
            False: Uses ''

    Returns:
        str: Complete URL in format: {base_url}/api/{endpoint}
            where base_url is determined by runtime environment:
            - Colab: _state['tunnel_url_1'], _state['tunnel_url_2']
            - Docker: _state['docker_url']
            - Local: _state['alltalk_protocol'] + _state['alltalk_ip_port']
    """
    debug_func_entry()
    print_message(
        f"\033[94mRunning on Docker is: \033[0m{_state['running_in_docker']}",
        message_type="debug_gradio_IP",
    )
    print_message(
        f"\033[94mRunning on Google is: \033[0m{_state['running_on_google_colab']}",
        message_type="debug_gradio_IP",
    )

    if _state['running_on_google_colab']:
        base_url = _state['tunnel_url_1']
    elif _state['running_in_docker']:
        base_url = _state['docker_url']
    else:
        base_url = f"{_state['alltalk_protocol']}{_state['alltalk_ip_port']}"

    prefix = "/api/" if include_api else "/"
    final_url = f"{base_url}{prefix}{endpoint.lstrip('/')}"
    print_message(
        f"\033[94mBase URL is set as  : \033[0m{final_url}",
        message_type="debug_gradio_IP",
    )
    return final_url


def build_dynamic_url(endpoint, include_protocol=True, include_api=True):
    """
    Build URL for endpoints that may need the current domain.
    Used when the URL needs to reference the actual server domain
    rather than predefined static URLs.

    Examples:
        - tts-generate
        - tts-generate-streaming
        - File URLs (with include_protocol=True)

    Args:
        endpoint (str): The API endpoint or path to append
        include_protocol (bool): Whether to include protocol prefix
            True: Uses 'protocol' global variable
            False: Uses 'http://'
        include_api (bool): Whether to include /api/ in the URL
            True: Uses '/api/'
            False: Uses ''

    Returns:
        str: Complete URL with format depending on environment:
            - Colab: _state['tunnel_url_1']/api/{endpoint}
            - Docker: _state['docker_url']/api/{endpoint}
            - Local: {protocol_prefix}{_state['my_current_url']}/api/{endpoint}
    """
    debug_func_entry()
    print_message(
        f"\033[94mRunning on Docker is: \033[0m{_state['running_in_docker']}",
        message_type="debug_gradio_IP",
    )
    print_message(
        f"\033[94mRunning on Google is: \033[0m{_state['running_on_google_colab']}",
        message_type="debug_gradio_IP",
    )

    prefix = "/api/" if include_api else "/"

    if _state['running_on_google_colab']:
        base_url = _state['tunnel_url_1']
    elif _state['running_in_docker']:
        base_url = _state['docker_url']
    else:
        protocol_prefix = f"{_state['alltalk_protocol']}" if include_protocol else "http://"
        base_url = f"{protocol_prefix}{_state['my_current_url']}"

    final_url = f"{base_url}{prefix}{endpoint.lstrip('/')}"
    print_message(
        f"\033[94mBase URL is set as  : \033[0m{final_url}",
        message_type="debug_gradio_IP",
    )
    return final_url


def update_docker_address(new_url):
    """
    Used within the gradio interface to update the IP/URL
    when running inside a docker environment.
    """
    _state['docker_url'] = new_url
    print_message(f"\033[94mDocker IP/URL for API set to: \033[0m{new_url}")
    return f"Docker IP/URL for API set to: {new_url}"


#################################################
# Pull all the settings from the AllTalk Server #
#################################################
def get_alltalk_settings():
    """Retrieves current settings from the AllTalk server including voices, models, and configuration.
    Returns a dictionary of settings with default values if server is unreachable.
    """
    debug_func_entry()

    # Define endpoints
    endpoints = {
        name: build_url(name) for name in ["voices", "rvcvoices", "currentsettings"]
    }

    # Default settings in case of failure
    default_settings = {
        "voices": ["Please Refresh Settings"],
        "rvcvoices": ["Please Refresh Settings"],
        "engines_available": ["Please Refresh Settings"],
        "current_engine_loaded": ["Please Refresh Settings"],
        "models_available": ["Please Refresh Settings"],
        "current_model_loaded": "Please Refresh Settings",
        "manufacturer_name": "",
        "deepspeed_capable": False,
        "deepspeed_available": False,
        "deepspeed_enabled": False,
        "generationspeed_capable": False,
        "generationspeed_set": 1.0,
        "lowvram_capable": False,
        "lowvram_enabled": False,
        "pitch_capable": False,
        "pitch_set": 0,
        "repetitionpenalty_capable": False,
        "repetitionpenalty_set": 10.0,
        "streaming_capable": False,
        "temperature_capable": False,
        "temperature_set": 0.75,
        "ttsengines_installed": False,
        "languages_capable": False,
        "multivoice_capable": False,
        "multimodel_capable": False,
    }

    try:
        # Make all requests with a single session for efficiency
        with requests.Session() as session:
            responses = {
                name: session.get(url, timeout=connection_timeout)
                for name, url in endpoints.items()
            }

        # Check if all responses are successful
        if not all(r.status_code == 200 for r in responses.values()):
            failed_endpoints = [
                name for name, r in responses.items() if r.status_code != 200
            ]
            print_message(
                f"Failed to retrieve data from: {', '.join(failed_endpoints)}",
                message_type="warning",
                component="TTS",
            )
            print_message(
                "Please confirm you have the correct IP/URL & that the system isnt currenly restarting",
                message_type="warning",
                component="TTS",
            )
            return default_settings

        # Parse responses
        voices_data = responses["voices"].json()
        rvcvoices_data = responses["rvcvoices"].json()
        settings_data = responses["currentsettings"].json()

        # Update state dictionary with all current values
        _state.update({
            'srv_current_voices': sorted(voices_data["voices"], key=lambda s: s.lower()),
            'srv_current_rvcvoices': rvcvoices_data["rvcvoices"],
            'srv_engines_available': sorted(settings_data["engines_available"]),
            'srv_current_engine_loaded': settings_data["current_engine_loaded"],
            'srv_models_available': sorted([model["name"] for model in settings_data["models_available"]]),
            'srv_current_model_loaded': settings_data["current_model_loaded"],
            'srv_character_voice': config.tgwui.tgwui_character_voice,
            'srv_narrator_voice': config.tgwui.tgwui_narrator_voice,
            'srv_rvc_character_voice': config.rvc_settings.rvc_char_model_file,
            'srv_rvc_narrator_voice': config.rvc_settings.rvc_narr_model_file,
            'srv_settings_capabilities': {
                'temperature_capable': settings_data["temperature_capable"],
                'repetitionpenalty_capable': settings_data["repetitionpenalty_capable"],
                'languages_capable': settings_data["languages_capable"],
                'generationspeed_capable': settings_data["generationspeed_capable"],
                'pitch_capable': settings_data["pitch_capable"],
                'streaming_capable': settings_data["streaming_capable"]
            }
        })

        # Construct and return settings dictionary
        base_settings = {
            "voices": _state['srv_current_voices'],
            "rvcvoices": _state['srv_current_rvcvoices'],
            "engines_available": _state['srv_engines_available'],
            "current_engine_loaded": _state['srv_current_engine_loaded'],
            "models_available": _state['srv_models_available'],
            "current_model_loaded": _state['srv_current_model_loaded'],
        }

        # Add remaining settings
        return {
            **base_settings,
            **{
                k: settings_data.get(k, v)
                for k, v in default_settings.items()
                if k not in base_settings
            }
        }

    except (RequestException, RequestsConnectionError) as e:
        print_message(
            f"Unable to connect to the {config.branding} API server: {str(e)}",
            message_type="warning",
            component="TTS",
        )
        print_message(
            "Please confirm you have the correct IP/URL & that the system isnt currenly restarting",
            message_type="warning",
            component="TTS",
        )
        return default_settings


# Pull all the current settings from the AllTalk server, if its online.
get_alltalk_settings()

#############################
#### TTS STOP GENERATION ####
#############################
def stop_generate_tts():
    """
    Sends a stop TTS generation request
    """
    debug_func_entry()
    api_url = build_url("stop-generation")
    try:
        stop_response = requests.put(api_url, timeout=connection_timeout)
        if stop_response.status_code == 200:
            return stop_response.json()["message"]
        print_message(
            f"Failed to stop generation. Status code:\n{stop_response.status_code}",
            message_type="warning",
        )
        return {"message": "Failed to stop generation"}
    except (RequestException, RequestsConnectionError) as e:
        print_message(
            f"Unable to connect to the {config.branding} server. Status code:\n{str(e)}",
            message_type="warning",
        )
        return {"message": "Failed to stop generation"}


def send_api_request(endpoint, payload=None, headers=None, params=None):
    """Base function for making API requests to the AllTalk server.

    Args:
        endpoint (str): API endpoint (e.g., '/api/reload')
        payload (dict, optional): POST data
        headers (dict, optional): Request headers
        params (dict, optional): URL parameters

    Returns:
        dict: Response data or error status
    """
    debug_func_entry()
    try:
        # Construct base URL based on environment
        base_url = build_url(endpoint.lstrip("/"))

        api_response = requests.post(
            base_url,
            json=payload,
            headers=headers or {"Content-Type": "application/json"},
            params=params,
            timeout=(5, 30)  # 5 sec connect, 30 sec read timeout
        )
        api_response.raise_for_status()

        return {
            "status": "success",
            "response": api_response.json() if api_response.content else None,
            "raw_response": api_response,
        }
    except RequestException as e:
        print_message(
            f"Error during request to webserver process: Status code:\n{e}",
            message_type="warning",
        )
        return {"status": "error", "message": str(e)}


def send_reload_request(value_sent):
    """Sends a request to reload the TTS model.

    Args:
        value_sent: The TTS method/model to load
    """
    debug_func_entry()
    params = {"tts_method": value_sent}
    return send_api_request("/api/reload", params=params)

#################################
#### Proxy Interface Manager ####
#################################
if _state.get('proxy_manager') is None:
    _state['proxy_manager'] = ProxyManager(AlltalkConfig)
    if config.proxy_settings.proxy_enabled and config.proxy_settings.start_on_startup:
        _state['proxy_manager'].start_proxy()
        
########################################
#### Whisper Transcription Handling ####
########################################
# pylint: disable=too-few-public-methods
class TranscriptionProgress:
    """Track progress of Whisper's audio transcription tasks including file counts and completion status."""
    def __init__(self):
        self.current_file = ""
        self.total_files = 0
        self.completed_files = 0
        self.current_progress = 0


whisper_progress = TranscriptionProgress()

# File size constants (in bytes)
WARNING_SIZE = 25 * 1024 * 1024  # 25MB
MAX_SIZE = 100 * 1024 * 1024  # 100MB

# Allowed audio formats
ALLOWED_AUDIO_FORMATS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".aac",
    ".wma",
    ".aiff",
    ".alac",
    ".opus",
}


def setup_directories(setup_script_dir):
    """Setup required directories"""
    debug_func_entry()
    base_dir = os.path.join(setup_script_dir, "transcriptions")
    uploads_dir = os.path.join(base_dir, "uploads")
    output_dir = os.path.join(base_dir, "output")

    for directory in [base_dir, uploads_dir, output_dir]:
        os.makedirs(directory, exist_ok=True)
        print_message(f"Created/verified directory: {directory}", "debug_transcribe")

    return base_dir, uploads_dir, output_dir


def validate_audio_file(file_path):
    """Validate file type and size"""
    debug_func_entry()
    file_size = os.path.getsize(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    mime_type = mimetypes.guess_type(file_path)[0]

    print_message(f"Validating file: {file_path}", "debug_transcribe")
    print_message(f"File size: {file_size/1024/1024:.1f}MB", "debug_transcribe")
    print_message(f"File extension: {file_ext}", "debug_transcribe")
    print_message(f"MIME type: {mime_type}", "debug_transcribe")

    validate_warnings = []
    errors = []

    if file_ext not in ALLOWED_AUDIO_FORMATS:
        msg = f"Invalid file format: {file_ext}"
        print_message(msg, "error")
        errors.append(msg)

    if mime_type and not mime_type.startswith("audio/"):
        msg = f"File does not appear to be audio: {mime_type}"
        print_message(msg, "error")
        errors.append(msg)

    if file_size > MAX_SIZE:
        msg = (
            f"File too large: {file_size/1024/1024:.1f}MB (max {MAX_SIZE/1024/1024}MB)"
        )
        print_message(msg, "error")
        errors.append(msg)
    elif file_size > WARNING_SIZE:
        msg = f"File is large ({file_size/1024/1024:.1f}MB) and may take longer to process"
        print_message(msg, "warning")
        validate_warnings.append(msg)

    return validate_warnings, errors


def cleanup_gpu_memory():
    """Thorough GPU memory cleanup"""
    debug_func_entry()
    if torch.cuda.is_available():
        # Clear PyTorch's CUDA cache
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
        # Clear any remaining CUDA memory
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print_message("GPU memory cleaned", "debug_transcribe")


def create_output_directory(base_dir, prefix=""):
    """Create a uniquely named output directory for this batch of transcriptions"""
    debug_func_entry()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}" if prefix else timestamp
    output_dir = os.path.join(base_dir, "output", dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print_message(f"Created output directory: {output_dir}", "debug_transcribe")
    return output_dir


def create_output_filename(input_path, output_dir, prefix=""):
    """Create an output filename based on the input path"""
    debug_func_entry()
    stem = Path(input_path).stem
    if prefix:
        return os.path.join(output_dir, f"{prefix}_{stem}")
    return os.path.join(output_dir, stem)


def delete_uploaded_files():
    """Delete all files in the uploads directory"""
    debug_func_entry()
    del_upload_script_dir = os.path.dirname(os.path.abspath(__file__))
    uploads_dir = os.path.join(del_upload_script_dir, "transcriptions", "uploads")
    deleted_count = 0

    print_message(f"Attempting to delete files in: {uploads_dir}", "debug_transcribe")

    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    deleted_count += 1
                    print_message(f"Deleted: {file}", "debug_transcribe")
            except FileNotFoundError:
                error_msg = f"File not found while trying to delete: {file}"
                print_message(error_msg, "error")
            except PermissionError:
                error_msg = f"Permission denied while trying to delete: {file}"
                print_message(error_msg, "error")
            except OSError as e:
                error_msg = f"OS error while deleting {file}: {str(e)}"
                print_message(error_msg, "error")

        msg = f"Deleted {deleted_count} uploaded audio files"
        print_message(msg)
        return msg

    msg = "No files to delete - uploads directory not found"
    print_message(msg, "warning")
    return msg


def create_srt_content(segments):
    """Convert whisper segments to SRT format"""
    debug_func_entry()
    srt_content = []
    for i, segment in enumerate(segments, start=1):
        # Convert start/end times to SRT format (HH:MM:SS,mmm)
        srt_start_time = timedelta(seconds=segment["start"])
        srt_end_time = timedelta(seconds=segment["end"])

        # Format with milliseconds
        start_str = str(srt_start_time).replace(".", ",")[:11]
        end_str = str(srt_end_time).replace(".", ",")[:11]

        # Ensure proper formatting for times less than 1 hour
        if len(start_str) < 11:
            start_str = "0" + start_str
        if len(end_str) < 11:
            end_str = "0" + end_str

        # Build SRT segment
        srt_content.append(
            f"{i}\n{start_str} --> {end_str}\n{segment['text'].strip()}\n"
        )

    return "\n".join(srt_content)


def process_audio_files(
    audio_files,
    model_size,
    output_format,
    delete_after,
    prefix="",
    gradio_progress=gr.Progress(track_tqdm=True), # pylint: disable=unused-argument  # Used by tqdm for progress tracking
):
    """Process multiple audio files and return paths to transcription files"""
    debug_func_entry()
    try:
        print_message(
            f"Starting transcription process with model: {model_size}",
            "debug_transcribe",
        )

        # Setup directories
        transcribe_script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir, uploads_dir, _ = setup_directories(transcribe_script_dir)

        # Create specific output directory for this batch
        output_dir = create_output_directory(base_dir, prefix)
        print_message(
            f"Using prefix: {prefix if prefix else 'none'}", "debug_transcribe"
        )

        # Validate and copy files to uploads directory
        processed_files = []
        validation_messages = []

        if not audio_files:
            msg = "No audio files provided for processing."
            print_message(msg, "error")
            return None, msg

        for audio_file in audio_files:
            print_message(f"Processing file: {audio_file}", "debug_transcribe")
            transcribe_warnings, errors = validate_audio_file(audio_file)
            file_name = Path(audio_file).name

            if errors:
                for error in errors:
                    validation_messages.append(f"Skipping {file_name}: {error}")
                continue

            validation_messages.extend(
                [f"Warning for {file_name}: {warning}" for warning in transcribe_warnings]
            )

            # Copy file to uploads directory
            upload_path = os.path.join(uploads_dir, file_name)
            shutil.copy2(audio_file, upload_path)
            print_message(f"Copied to uploads: {upload_path}", "debug_transcribe")
            processed_files.append(upload_path)

        if not processed_files:
            msg = "No valid files to process\n" + "\n".join(validation_messages)
            print_message(msg, "error")
            return None, msg

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print_message(
            f"Loading Whisper model {model_size} on {device}", "debug_transcribe"
        )
        _state['whisper_model'] = whisper.load_model(model_size, device=device)

        output_files = []
        metadata = []

        whisper_progress.total_files = len(processed_files)

        for idx, audio_file in enumerate(
            tqdm(processed_files, desc="Processing files")
        ):
            try:
                whisper_progress.current_file = Path(audio_file).name
                whisper_progress.completed_files = idx
                print_message(f"Transcribing: {audio_file}", "debug_transcribe")

                # Transcribe audio
                result = _state['whisper_model'].transcribe(audio_file)

                # Create output file
                base_output_path = create_output_filename(
                    audio_file, output_dir, prefix
                )

                # Save based on format
                if output_format == "txt":
                    output_path = f"{base_output_path}.txt"
                    with open(output_path, "w", encoding="utf-8") as file:
                        file.write(result["text"])
                elif output_format == "srt":
                    output_path = f"{base_output_path}.srt"
                    with open(output_path, "w", encoding="utf-8") as file:
                        file.write(create_srt_content(result["segments"]))
                else:  # json
                    output_path = f"{base_output_path}.json"
                    with open(output_path, "w", encoding="utf-8") as file:
                        json.dump(result, file, indent=4, ensure_ascii=False)

                print_message(
                    f"Saved transcription to: {output_path}", "debug_transcribe"
                )
                output_files.append(output_path)

                # Collect metadata
                metadata.append(
                    {
                        "original_file": Path(audio_file).name,
                        "duration": (
                            result["segments"][-1]["end"] if result["segments"] else 0
                        ),
                        "output_file": Path(output_path).name,
                        "model_used": model_size,
                        "format": output_format,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Delete audio file if requested
                if delete_after:
                    os.unlink(audio_file)
                    print_message(
                        f"Deleted audio file: {audio_file}", "debug_transcribe"
                    )

            except (whisper.RuntimeError, torch.cuda.OutOfMemoryError) as e:
                error_msg = f"Error in transcription of {Path(audio_file).name}: {str(e)}"
                print_message(error_msg, "error")
                validation_messages.append(error_msg)
                continue
            except (OSError, IOError) as e:
                error_msg = f"File error processing {Path(audio_file).name}: {str(e)}"
                print_message(error_msg, "error")
                validation_messages.append(error_msg)
                continue
            except (ValueError, TypeError) as e:
                error_msg = f"Data processing error in {Path(audio_file).name}: {str(e)}"
                print_message(error_msg, "error")
                validation_messages.append(error_msg)
                continue

        # Create summary file
        summary_path = os.path.join(
            output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "processed_files": metadata,
                    "total_files": len(processed_files),
                    "successful_transcriptions": len(output_files),
                    "processing_date": datetime.now().isoformat(),
                    "validation_messages": validation_messages,
                },
                f,
                indent=4,
            )

        print_message(f"Created summary file: {summary_path}", "debug_transcribe")

        # Create ZIP file containing all outputs
        zip_path = os.path.join(
            base_dir,
            "output",
            (
                f"{prefix}_transcriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                if prefix
                else f"transcriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            ),
        )
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in output_files + [summary_path]:
                # Preserve the directory structure in the ZIP
                arcname = os.path.relpath(file, os.path.dirname(output_dir))
                zipf.write(file, arcname)

        print_message(f"Created ZIP file: {zip_path}", "debug_transcribe")

        status_message = "\n".join(
            validation_messages
            + [
                f"Processed {m['original_file']}: {m['duration']:.2f} seconds"
                for m in metadata
            ]
        )
        status_message += f"\nFiles saved in: {output_dir}"

        return zip_path, status_message

    except (OSError, IOError) as e:
        print_message(f"File system error: {str(e)}", "error")
        return None, f"File system error: {str(e)}"
    except torch.cuda.OutOfMemoryError as e:
        print_message(f"GPU memory error: {str(e)}", "error")
        return None, "GPU memory error: Consider using a smaller model"
    except ValueError as e:
        print_message(f"Invalid input error: {str(e)}", "error")
        return None, f"Configuration error: {str(e)}"
    finally:
        # Cleanup GPU memory
        if 'whisper_model' in _state:
            del _state['whisper_model']
        cleanup_gpu_memory()


########################
#### Live Dictation ####
########################
def reset_audio_stream_handlers(dictate_audio, state, text_output, audio_plot):
    """Reset handlers for the dictation audio input."""
    if not isinstance(dictate_audio, gr.Audio):
        print_message("Error: 'dictate_audio' is not a valid Gradio Audio component.", "error")
        return

    # Clear existing handlers
    if hasattr(dictate_audio, "stop_recording"):
        dictate_audio.stop_recording(fn=None)
    if hasattr(dictate_audio, "start_recording"):
        dictate_audio.start_recording(fn=None)
    if hasattr(dictate_audio, "stream"):
        dictate_audio.stream(fn=None)

    # Re-add the handlers # Ignore them being undefined
    dictate_audio.start_recording(
        fn=on_start_recording, inputs=[state], outputs=[state] # pylint: disable=undefined-variable
    ).success(fn=None, js="() => {console.log('Recording started');}")

    dictate_audio.stop_recording(
        fn=on_stop_recording, inputs=[state], outputs=[state] # pylint: disable=undefined-variable
    ).success(fn=None, js="() => {console.log('Recording stopped');}")

    dictate_audio.stream(
        fn=process_audio,
        inputs=[dictate_audio, state],
        outputs=[state, text_output, audio_plot],
        show_progress=False,
        concurrency_limit=1,
    )


def setup_transcription_directory():
    """Setup directory for saving transcriptions"""
    debug_func_entry()
    trans_script_dir = os.path.dirname(os.path.abspath(__file__))
    transcripts_dir = os.path.join(trans_script_dir, "transcriptions", "live_dictation")
    os.makedirs(transcripts_dir, exist_ok=True)
    return transcripts_dir


def create_transcript_file(directory, prefix=""):
    """Create a new transcript file with timestamp"""
    debug_func_entry()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{prefix}_dictation_{timestamp}.txt"
        if prefix
        else f"dictation_{timestamp}.txt"
    )
    return os.path.join(directory, filename)


def load_whisper_model(model_name):
    """Load the Whisper model"""
    debug_func_entry()
    print_message(f"Loading Whisper model: {model_name}", "debug_transcribe")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _state['whisper_model'] = whisper.load_model(model_name, device=device)
    print_message(f"Model loaded on {device}", "debug_transcribe")
    return _state['whisper_model']


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a butterworth bandpass filter"""
    debug_func_entry()
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    return b, a


def apply_audio_processing(audio_data, sample_rate, settings):
    """Apply audio processing based on settings"""
    debug_func_entry()
    if not settings.get("enable_audio_processing"):
        return audio_data

    # Convert to float32 and normalize
    audio_data = audio_data.astype(np.float32)

    if settings.get("bandpass_filter"):
        # Apply bandpass filter
        b, a = butter_bandpass(
            settings.get("bandpass_low", 85),
            settings.get("bandpass_high", 3800),
            sample_rate,
        )
        audio_data = scipy_signal.filtfilt(b, a, audio_data)

    if settings.get("noise_reduction"):
        # Simple noise reduction
        noise_floor = np.mean(np.abs(audio_data)) * 2
        audio_data[np.abs(audio_data) < noise_floor] = 0

    if settings.get("compression"):
        # Apply compression
        threshold = 0.1
        ratio = 0.5
        makeup_gain = 1.5

        mask = np.abs(audio_data) > threshold
        audio_data[mask] = np.sign(audio_data[mask]) * (
            threshold + (np.abs(audio_data[mask]) - threshold) * ratio
        )
        audio_data *= makeup_gain

    # Final normalization
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val

    return audio_data


def identify_speaker(segments):
    """Simple speaker identification based on segment characteristics"""
    debug_func_entry()
    # Very basic approach - could be enhanced with proper speaker diarization
    current_speaker = 1
    last_end_time = 0

    for segment in segments:
        # If there's a significant gap, might be a different speaker
        if segment["start"] - last_end_time > 1.0:
            current_speaker = 2 if current_speaker == 1 else 1
        last_end_time = segment["end"]

    return current_speaker

def visualize_audio_levels(
    audio_data,
    sample_rate=16000,  # Default sample rate for Whisper
    window_size=1024,
    stride=None,
    y_axis_range=(0, 1),  # Fixed range for audio levels
):
    """Create an enhanced audio level visualization."""
    debug_func_entry()

    # Set stride for overlapping windows
    stride = stride or window_size // 2  # Default: 50% overlap

    # Calculate RMS values for each window
    windows = range(0, len(audio_data) - window_size, stride)
    rms_values = [
        np.sqrt(np.mean(audio_data[i : i + window_size] ** 2)) for i in windows
    ]

    # Calculate time points in seconds
    time_points = [(i * stride) / sample_rate for i in range(len(rms_values))]

    # Downsample if too many points (performance optimization)
    max_points = 1000
    if len(rms_values) > max_points:
        step = len(rms_values) // max_points
        rms_values = rms_values[::step]
        time_points = time_points[::step]

    # Create the plot
    fig = go.Figure()

    # Add audio levels line
    fig.add_trace(
        go.Scatter(x=time_points, y=rms_values, mode="lines", name="Audio Level")
    )

    # Add optional dynamic range visualization (e.g., warning thresholds)
    safe_level = 0.7  # Example threshold for 'safe' audio level
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=[safe_level] * len(time_points),
            mode="lines",
            line={"dash": "dash", "color": "red"},
            name="Warning Threshold",
        )
    )

    # Update layout
    fig.update_layout(
        title="Real-time Audio Levels",
        xaxis_title="Time (s)",
        yaxis_title="Level",
        template="plotly_white",  # Use a clean white theme
        xaxis={"showgrid": True},
        yaxis={
            "showgrid": True,
            "range": y_axis_range,  # Set fixed range for audio levels
        },
    )
    return fig


def process_audio(audio, state):
    """Process audio chunk and update transcript"""
    debug_func_entry()
    # Default return values for error cases
    empty_plot = visualize_audio_levels(np.zeros(1024))
    default_return = (
        state if state else None,
        state["display_text"] if state and "display_text" in state else "",
        empty_plot,
    )

    if audio is None or state is None or not state.get("is_active"):
        return default_return

    try:
        # Get audio data and sample rate
        sample_rate, audio_data = audio[0], audio[1]

        # Convert to float32 and normalize
        audio_data = audio_data.astype(np.float32) / 32768.0

        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_data**2))
        print_message(f"Audio energy level: {energy}", "debug_transcribe")

        # Skip if too quiet
        if energy < 0.005:
            print_message("Audio too quiet, skipping", "debug_transcribe")
            plot_data = visualize_audio_levels(audio_data)
            return state, state["display_text"], plot_data

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            print_message(f"Resampling from {sample_rate} to 16000", "debug_transcribe")
            number_of_samples = round(len(audio_data) * 16000 / sample_rate)
            audio_data = scipy_signal.resample(audio_data, number_of_samples).astype(
                np.float32
            )

        # Simple normalization
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        # pylint: disable=line-too-long
        print_message(
            f"Processing audio chunk - shape: {audio_data.shape}, range: [{audio_data.min():.3f}, {audio_data.max():.3f}]",
            "debug_transcribe",
        )

        if 'whisper_model' not in _state:
            error_msg = "Whisper model not loaded"
            print_message(error_msg, "error")
            return default_return

        # Process with Whisper
        result = _state['whisper_model'].transcribe(
            audio_data,
            language=state.get("source_language", "auto"),
            task="translate" if state.get("translate_to_english") else "transcribe",
            temperature=0.0,
            condition_on_previous_text=True,
        )

        transcribed_text = result["text"].strip()

        if transcribed_text:
            # Format text based on timestamps setting
            if state.get("add_timestamps"):
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_text = f"[{timestamp}] {transcribed_text}"
                state["text"] += formatted_text + "\n"
            # For non-timestamped text, append with space and handle sentence endings
            if state["text"] and not state["text"].endswith((".", "!", "?", "\n")):
                state["text"] += " "
            state["text"] += transcribed_text

            # Update display text
            state["display_text"] = state["text"]

            # Get confidence scores (if available)
            try:
                confidences = [
                    segment.get("confidence", 1.0) for segment in result["segments"]
                ]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 1.0
                )
                state["display_text"] += f"\n\nConfidence: {avg_confidence:.2%}"
            except (KeyError, TypeError, AttributeError):
                pass  # Non-critical confidence calculation failed

            # Save transcript
            save_transcript(state, state.get("export_format", "txt"))

            # Create audio plot
            try:
                plot_data = visualize_audio_levels(audio_data)
            except (ValueError, TypeError, np.VisualWarning) as plot_error:
                print_message(
                    f"Plot error (non-critical): {plot_error}", "debug_transcribe"
                )
                plot_data = None

            return state, state["display_text"], plot_data
        return default_return

    except (ValueError, TypeError) as e:
        error_msg = f"Error processing audio data: {str(e)}"
    except np.VisualWarning as e:
        error_msg = f"Error in numerical operations: {str(e)}"
    except RuntimeError as e:
        error_msg = f"Error in speech recognition: {str(e)}"
    except (OSError, IOError) as e:
        error_msg = f"Error saving transcript: {str(e)}"
    print_message(error_msg, "error")
    return default_return

def create_srt_file(segments, filename):
    """Create an SRT file from segments"""
    debug_func_entry()
    srt_content = create_srt_content(segments)
    with open(filename, "w", encoding="utf-8") as open_srt_f:
        open_srt_f.write(srt_content)

def save_transcript(state, format_type):
    """Save transcript in specified format"""
    debug_func_entry()
    if not state or not state.get("text"):
        return

    # Format text based on timestamps setting
    if state.get("add_timestamps"):
        save_text = state["text"]  # Already formatted with timestamps
    else:
        # Format as flowing text with proper sentence spacing
        sentences = state["text"].replace("\n", " ").split(". ")
        save_text = ". ".join(s.strip() for s in sentences if s.strip())
        if not save_text.endswith((".", "!", "?")):
            save_text += "."

    if format_type == "txt":
        with open(state["output_file"], "w", encoding="utf-8") as save_transcript_f:
            save_transcript_f.write(save_text)

    elif format_type == "srt":
        srt_filename = state["output_file"].replace(".txt", ".srt")
        create_srt_file(state.get("segments", []), srt_filename)

    elif format_type == "json":
        json_filename = state["output_file"].replace(".txt", ".json")
        metadata = {
            "text": save_text,
            "segments": state.get("segments", []),
            "word_count": len(save_text.split()),
            "model": state["model_name"],
            "settings": state["settings"],
        }
        with open(json_filename, "w", encoding="utf-8") as save_transcript_f:
            json.dump(metadata, save_transcript_f, indent=2)

    print_message(f"Saved transcript to: {state['output_file']}", "debug_transcribe")


def start_new_dictation(
    model_name,
    prefix,
    language,
    source_lang,
    translate,
    export_fmt,
    timestamps,
    diarization,
    audio_proc,
    bandpass,
    noise_red,
    compress,
    silence_thresh,
    bp_low,
    bp_high,
):
    """Load model and prepare for dictation"""
    debug_func_entry()
    try:
        # Load model into state
        load_whisper_model(model_name)

        transcripts_dir = setup_transcription_directory()
        output_file = create_transcript_file(transcripts_dir, prefix)

        state = {
            "text": "",
            "display_text": "",
            "output_file": output_file,
            "chunks": 0,
            "is_active": True,
            "model_name": model_name,
            "word_count": 0,
            # Settings
            "language": language,
            "source_language": source_lang,
            "translate_to_english": translate,
            "export_format": export_fmt,
            "add_timestamps": timestamps,
            "enable_diarization": diarization,
            "settings": {
                "enable_audio_processing": audio_proc,
                "bandpass_filter": bandpass,
                "noise_reduction": noise_red,
                "compression": compress,
                "silence_threshold": silence_thresh,
                "bandpass_low": bp_low,
                "bandpass_high": bp_high,
            },
        }

        print_message(f"Started new transcription: {output_file}", "debug_transcribe")
        message = "Model loaded! Click the microphone icon to start/stop recording."

        return (
            state,
            message,
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True),
        )
    except (whisper.RuntimeError, torch.cuda.OutOfMemoryError) as e:
        error_msg = f"Model error: {str(e)}"
        return_msg = "Error: Model could not be loaded - try a smaller model"
    except (OSError, IOError) as e:
        error_msg = f"File system error: {str(e)}"
        return_msg = "Error: Could not create necessary files/directories"
    except ValueError as e:
        error_msg = f"Invalid parameter: {str(e)}"
        return_msg = "Error: Invalid configuration parameters provided"
    print_message(error_msg, "error")
    return (
        None,
        return_msg,
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
    )

def finish_dictation(state, dictate_audio, text_output, audio_plot):
    """End dictation and cleanup."""
    debug_func_entry()

    if state is not None:
        print_message(f"Finished dictation: {state.get('output_file', 'Unknown')}", "debug_transcribe")
        save_transcript(state, state.get("export_format", "txt"))
        state["is_active"] = False

    # Reset the audio stream handlers only if dictate_audio is a valid component
    if isinstance(dictate_audio, gr.Audio):
        reset_audio_stream_handlers(dictate_audio, state, text_output, audio_plot)

    # Clean up the model
    if 'whisper_model' in _state:
        _state['whisper_model'].to("cpu")  # Move model to CPU first
        del _state['whisper_model']

    # Clean up GPU resources
    cleanup_gpu_memory()
    torch.cuda.synchronize()

    return (
        state,
        "Transcription finished. Ready for a new session!",
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


##################################################################
#     _    _ _ _____     _ _       ____               _ _        #
#    / \  | | |_   _|_ _| | | __  / ___|_ __ __ _  __| (_) ___   #
#   / _ \ | | | | |/ _` | | |/ / | |  _| '__/ _` |/ _` | |/ _ \  #
#  / ___ \| | | | | (_| | |   <  | |_| | | | (_| | (_| | | (_) | #
# /_/   \_\_|_| |_|\__,_|_|_|\_\  \____|_|  \__,_|\__,_|_|\___/  #
#                                                                #
##################################################################

if gradio_enabled is True:
    # Get the directory of the current script
    script_dir = Path(__file__).resolve().parent
    # at_default_voice_gr = config.tgwui.tgwui_character_voice # Disabled as I think now un-used
    ####################################################
    # Dynamically import the Themes builder for Gradio #
    ####################################################
    themesmodule_path = (
        this_dir / "system" / "gradio_pages" / "themes" / "loadThemes.py"
    )
    # Add the directory containing the module to the system path
    sys.path.insert(0, str(this_dir / "system" / "gradio_pages" / "themes"))
    # Import the module dynamically
    spec = importlib.util.spec_from_file_location("loadThemes", themesmodule_path)
    loadThemes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loadThemes)
    # Load the theme list from JSON file
    theme_list = loadThemes.get_list()
    # Load the selected theme from configuration
    selected_theme = loadThemes.load_json()
    ###########################################################
    # Finish Dynamically import the Themes builder for Gradio #
    ###########################################################
    # Adjust the module search path based on the execution context
    if script_dir.name == "alltalk_tts":
        # Standalone execution: Add the script directory to the module search path
        sys.path.insert(0, str(script_dir))
    else:
        # Running as part of text-generation-webui: Add the parent directory to the module search path
        sys.path.insert(0, str(script_dir.parent))

    # Determine if running as standalone or within another project
    if __name__ == "__main__":
        # Determine if running as standalone or within another project
        if "text-generation-webui" not in script_dir.parts:
            # Standalone execution
            base_package = None  # No base package needed for absolute imports
        else:
            # Running within text-generation-webui
            # Dynamically build the base package using the current folder name
            current_folder_name = script_dir.name
            base_package = f"extensions.{current_folder_name}"

    def dynamic_import(module_path, package=None):
        """
        Dynamically import a module from a given path.

        Args:
            module_path (str): Path or name of module to import
            package (str, optional): Package to import module from. Defaults to None.

        Returns:
            module: Imported module object or None if import fails
        """
        debug_func_entry()
        try:
            if package:
                new_module = importlib.import_module(module_path, package=package)
            else:
                new_module = importlib.import_module(module_path)
            return new_module
        except ModuleNotFoundError as e:
            print_message(
                f"Module not found: {module_path} - {str(e)}", message_type="error"
            )
            return None
        except (ImportError, SyntaxError) as e:
            print_message(f"Error importing {module_path}: {str(e)}", message_type="error")
            return None

    def load_engine_configs(_state):
        """Load engine configs into globals for Gradio settings pages"""
        for engine_name in _state.get('srv_engines_available', []):
            if base_package:
                module_name = f"{base_package}.system.tts_engines.{engine_name}.{engine_name}_settings_page"
            else:
                module_name = f"system.tts_engines.{engine_name}.{engine_name}_settings_page"

            module = dynamic_import(module_name, base_package)
            if module:
                # Load the engine's config from its JSON file
                json_file_path = os.path.join(this_dir, "system", "tts_engines", engine_name, "model_settings.json")
                try:
                    with open(json_file_path, "r", encoding="utf-8") as config_file:
                        globals()[f"{engine_name}_model_config_data"] = json.load(config_file)
                except FileNotFoundError:
                    print_message(f"Could not find settings file for {engine_name}")
                except json.JSONDecodeError:
                    print_message("Invalid JSON in settings file for {engine_name}")

    # Call this before creating the Gradio interface to dynamically import TTS Engine pages
    load_engine_configs(_state)

    ###########################################
    # Finishing Dynamically importing Modules #
    ###########################################

    ##########################################################################################
    # Pulls the current AllTalk Server settings & updates gradio when Refresh button pressed #
    ##########################################################################################
    def at_update_dropdowns():
        """Update the gradio dropdowns with the current settings from state"""
        debug_func_entry()

        # Call get_alltalk_settings to update state with latest values
        get_alltalk_settings()

        # Build stream choices if streaming is available
        if _state['srv_settings_capabilities']['streaming_capable']:
            gen_choices = [("Standard", "false"), ("Streaming (Disable Narrator)", "true")]
        else:
            gen_choices = [("Standard", "false")]

        # Set language label based on capability
        language_label = "Languages" if _state[
            'srv_settings_capabilities'
            ][
                'languages_capable'
                ] else "Model not multi language"

        # Handle default voices if they're not in available voices
        current_character_voice = _state['srv_character_voice']
        if current_character_voice not in _state['srv_current_voices']:
            current_character_voice = _state['srv_current_voices'][0] if _state['srv_current_voices'] else ""

        current_narrator_voice = _state['srv_narrator_voice']
        if current_narrator_voice not in _state['srv_current_voices']:
            current_narrator_voice = _state['srv_current_voices'][0] if _state['srv_current_voices'] else ""

        rvc_character_voice = _state['srv_rvc_character_voice']
        if rvc_character_voice not in _state['srv_current_rvcvoices']:
            rvc_character_voice = _state['srv_current_rvcvoices'][0] if _state['srv_current_rvcvoices'] else ""

        rvc_narrator_voice = _state['srv_rvc_narrator_voice']
        if rvc_narrator_voice not in _state['srv_current_rvcvoices']:
            rvc_narrator_voice = _state['srv_current_rvcvoices'][0] if _state['srv_current_rvcvoices'] else ""

        # Return all Gradio component updates
        return (
            gr.Dropdown(choices=gen_choices, interactive=True),
            gr.Dropdown(choices=_state['srv_current_voices'], value=current_character_voice, interactive=True),
            gr.Dropdown(choices=_state['srv_current_rvcvoices'], value=rvc_character_voice, interactive=True),
            gr.Dropdown(choices=_state['srv_current_voices'], value=current_narrator_voice, interactive=True),
            gr.Dropdown(choices=_state['srv_current_rvcvoices'], value=rvc_narrator_voice, interactive=True),
            gr.Slider(interactive=_state['srv_settings_capabilities']['generationspeed_capable']),
            gr.Slider(interactive=_state['srv_settings_capabilities']['pitch_capable']),
            gr.Slider(interactive=_state['srv_settings_capabilities']['temperature_capable']),
            gr.Slider(interactive=_state['srv_settings_capabilities']['repetitionpenalty_capable']),
            gr.Dropdown(interactive=_state['srv_settings_capabilities']['languages_capable'], label=language_label),
            gr.Dropdown(choices=_state['srv_models_available'], value=_state['srv_current_model_loaded']),
            gr.Dropdown(choices=_state['srv_engines_available'], value=_state['srv_current_engine_loaded'])
        )

    ######################################################################################
    # Sends request to reload the current TTS engine & set the default TTS engine loaded #
    ######################################################################################
    def set_engine_loaded(chosen_engine_name):
        """
        Send an engine reload request and update tts_engines list with the chosen engine
        and selected model.

        Args:
            chosen_engine_name (str): Name of the engine to load

        Returns:
            tuple: Success message and updated UI elements
            
        Raises:
            RequestsConnectionError: If unable to connect to TTS engine
            TimeoutError: If engine doesn't become ready within retry limit
        """
        debug_func_entry()

        # Use the config system to update the engine
        tts_engines_config_loaded = AlltalkTTSEnginesConfig.get_instance()
        # Update the engine and selected model
        if tts_engines_config_loaded.is_valid_engine(chosen_engine_name):
            # Find the engine and its selected_model in engines_available
            for engine in tts_engines_config.engines_available:
                if engine.name == chosen_engine_name:
                    # Update both engine_loaded and selected_model at top level
                    tts_engines_config_loaded.engine_loaded = engine.name
                    tts_engines_config_loaded.selected_model = engine.selected_model
                    break

            # Change engine in config - this updates selected_model automatically
            tts_engines_config_loaded.change_engine(chosen_engine_name)
            tts_engines_config_loaded.save()

            # Update state
            _state['srv_current_engine_loaded'] = chosen_engine_name
            _state['srv_current_model_loaded'] = tts_engines_config.selected_model

        # Save changes
        tts_engines_config_loaded.save()

        # Restart the subprocess
        restart_subprocess()

        # Wait for the engine to be ready with error handling and retries
        max_retries = 80
        retry_delay = 1  # seconds
        retries = 0
        while retries < max_retries:
            try:
                ready_response = requests.get(url, timeout=(5, 30))
                if ready_response.status_code == 200 and ready_response.text == "Ready":
                    break
            except RequestException:
                pass

            retries += 1
            if retries == max_retries:
                raise RequestsConnectionError(
                    "Failed to connect to the TTS engine after maximum retries. "
                    f"Attempted {max_retries} times over {max_retries * retry_delay} seconds."
                )

            time.sleep(retry_delay)

        # Update the dropdowns directly
        print_message("")
        print_message("Server Ready")

        # Update dropdowns - this line would change
        return_values = at_update_dropdowns()

        return (
            "TTS Engine changed successfully!",  # Output message
            *return_values,  # Unpack all the UI update values
        )

    ###############################
    # Sends voice2rvc request off #
    ###############################
    def voice2rvc(audio, rvc_voice, rvc_pitch, rvc_f0method):
        """
        Convert voice audio using RVC (Retrieval-based Voice Conversion).

        Args:
            audio: Input audio (tuple of sample_rate and data, or file path)
            rvc_voice: Selected RVC voice model
            rvc_pitch: Pitch adjustment value
            rvc_f0method: F0 extraction method

        Returns:
            str: Path to converted audio file, or None if conversion fails
        """
        debug_func_entry()
        # Save the uploaded or recorded audio to a file
        input_tts_path = this_dir / "outputs" / "voice2rvcInput.wav"

        if rvc_voice == "Disabled":
            print_message("Voice2RVC Convert: No RVC voice was selected")
            return None

        if audio is None:
            print_message("Voice2RVC Convert: No recorded audio was provided")
            return None

        if isinstance(audio, tuple):
            sample_rate, audio_data = audio
            # Save the numpy array as a wav file
            sf.write(input_tts_path, audio_data, sample_rate)
        else:
            # It's a file path
            os.rename(audio, input_tts_path)

        # Define the output path for the processed audio
        output_rvc_path = this_dir / "outputs" / "voice2rvcOutput.wav"
        voice2rvc_url = build_url("voice2rvc")

        # Submit the paths to the API endpoint
        voice2rvc_response = requests.post(
                voice2rvc_url,
                data={
                    "input_tts_path": str(input_tts_path),
                    "output_rvc_path": str(output_rvc_path),
                    "pth_name": rvc_voice,
                    "pitch": rvc_pitch,
                    "method": str(rvc_f0method),
                },
                timeout=(5, 30)  # 5 sec connect timeout, 30 sec read timeout
            )

        if voice2rvc_response.status_code == 200:
            result = voice2rvc_response.json()
            if result["status"] == "success":
                return result["output_path"]

        return None

    ##################################################################################################
    # Sends request to reload the current TTS engine model & set the default TTS engine model loaded #
    ##################################################################################################
    def change_model_loaded(new_engine_name, new_selected_model):
        """
        Change the currently loaded TTS model and update configuration.

        Args:
            new_engine_name (str): Name of TTS engine to use
            new_selected_model (str): Name of model to load

        Returns:
            tuple: Status message and updated UI component values, or
            dict: Error status and message if request fails
        """
        debug_func_entry()
        try:
            print_message("")
            print_message("\033[94mChanging model loaded. Please wait.\033[00m")
            print_message("")
            model_loaded_url = build_url("reload")
            payload = {"tts_method": new_selected_model}
            model_loaded_response = requests.post(model_loaded_url, params=payload, timeout=(5, 30))
            model_loaded_response.raise_for_status()  # Raises an HTTPError for bad responses
            # Update the tts_engines.json file
            tts_engines_file = os.path.join(
                this_dir, "system", "tts_engines", "tts_engines.json"
            )
            with open(tts_engines_file, "r", encoding="utf-8") as change_model_f:
                tts_engines_data = json.load(change_model_f)
            tts_engines_data["selected_model"] = f"{new_selected_model}"
            for engine in tts_engines_data["engines_available"]:
                if engine["name"] == new_engine_name:
                    engine["selected_model"] = f"{new_selected_model}"
                    break
            with open(tts_engines_file, "w", encoding="utf-8") as change_model_f:
                json.dump(tts_engines_data, change_model_f)
            print_message("")
            print_message("Server Ready")

            return_values = at_update_dropdowns()

            return ("TTS Model changed successfully!", *return_values)

        except RequestException as e:
            # Handle the HTTP request error
            print_message(
                f"Error during request to webserver process: Status code:\n{e}",
                message_type="warning",
            )
            return {"status": "error", "message": str(e)}

    debugging_options = config.debugging
    debugging_choices = list(vars(debugging_options).keys())
    default_values = [key for key, value in vars(debugging_options).items() if value]

    def generate_tts(
        gen_text,
        gen_char,
        rvcgen_char,
        rvcgen_char_pitch,
        gen_narr,
        rvcgen_narr,
        rvcgen_narr_pitch,
        gen_narren,
        gen_textni,
        gen_repetition,
        gen_lang,
        gen_filter,
        gen_speed,
        gen_pitch,
        gen_autopl,
        gen_autoplvol,
        gen_filen,
        gen_temperature,
        gen_filetime,
        gen_stream,
        gen_stopcurrentgen,
    ):
        """
        Send TTS request from the gradio interface to the API address
        """
        debug_func_entry()
        api_url = build_dynamic_url("tts-generate", include_protocol=False)
        if api_url == "http://null/api/tts-generate":  # Check if api_url is null
            print_message(
                "The URL in the API request was not set. This is usually because of a TTS request from",
                message_type="error",
            )
            print_message(
                "the Gradio web interface when the AllTalk server has been resarted but the Gradio page",
                message_type="error",
            )
            print_message(
                "was not re-loaded. Please refresh your Gradio page and try again.",
                message_type="error",
            )
            # pylint: disable=line-too-long
            return None, str(
                "Error: Did you restart the AllTalk Server and not refresh your Gradio page? Please see the console/terminal message."
            )
        if gen_text == "":
            print_message("No Text was sent to generate as TTS")
            return None, str("No Text was sent to generate as TTS")
        if gen_stopcurrentgen:
            stop_generate_tts()
        if gen_stream == "true":
            api_url = build_dynamic_url(
                "tts-generate-streaming", include_protocol=False
            )
            encoded_text = requests.utils.quote(gen_text)
            # pylint: disable=line-too-long
            streaming_url = f"{api_url}?text={encoded_text}&voice={gen_char}&language={gen_lang}&output_file={gen_filen}"
            return streaming_url, str("TTS Streaming Audio Generated")
        # pylint: disable=line-too-long
        tts_data = {
            "text_input": gen_text,
            "text_filtering": gen_filter,
            "character_voice_gen": gen_char,
            "rvccharacter_voice_gen": rvcgen_char,
            "rvccharacter_pitch": rvcgen_char_pitch,
            "narrator_enabled": str(gen_narren).lower(),
            "narrator_voice_gen": gen_narr,
            "rvcnarrator_voice_gen": rvcgen_narr,
            "rvcnarrator_pitch": rvcgen_narr_pitch,
            "text_not_inside": gen_textni,
            "language": gen_lang,
            "output_file_name": gen_filen,
            "output_file_timestamp": str(gen_filetime).lower(),
            "autoplay": str(gen_autopl).lower(),
            "autoplay_volume": str(gen_autoplvol),
            "speed": str(gen_speed),
            "pitch": str(gen_pitch),
            "temperature": str(gen_temperature),
            "repetition_penalty": str(gen_repetition),
        }
        print_message(
            "\033[94mDebug of data > def generate_tts > script.py\033[0m",
            message_type="debug_tts_variables",
        )

        keys = list(tts_data.keys())
        for i, key in enumerate(keys):
            # Use └─ for the last item, and ├─ for others
            prefix = "└─" if i == len(keys) - 1 else "├─"
            print_message(f"{prefix} {key}: {tts_data[key]}", message_type="debug_tts_variables")
        print_message(
            f"API Url being used: {api_url}", message_type="debug_tts_variables"
        )
        max_retries = 180  # 3 minutes
        retry_delay = 1  # seconds
        retries = 0
        while retries < max_retries:
            try:
                tts_response = requests.post(api_url, data=tts_data, timeout=60)
                tts_response.raise_for_status()
                result = tts_response.json()

                if gen_autopl == "true":
                    return None, str("TTS Audio Generated (Played remotely)")
                if config.api_def.api_use_legacy_api:
                    return result["output_file_url"], str("TTS Audio Generated")

                # Prepend the URL and PORT to the output_file_url
                output_file_url = build_dynamic_url(
                    result["output_file_url"].lstrip("/"),
                    include_protocol=False,
                    include_api=False,
                )
                return output_file_url, str("TTS Audio Generated")

            except (requests.exceptions.Timeout, json.JSONDecodeError, RequestException) as e:
                retries += 1
                error_message = None
                if retries == max_retries:
                    if isinstance(e, requests.exceptions.Timeout):
                        error_message = "Request timed out after maximum retries"
                    elif isinstance(e, json.JSONDecodeError):
                        error_message = "Failed to parse API response"
                        print_message(f"Error Details: {str(e)}", message_type="error")
                        print_message(f"Raw response: {tts_response.content}", message_type="error")
                    elif isinstance(e, RequestException):
                        error_message = "An error occurred while communicating with the API"
                        print_message(f"Error Details: {str(e)}", message_type="error")
                    print_message(f"Error: {error_message}", message_type="error")
                    return None, str(error_message)
                if isinstance(e, RequestException):
                    time.sleep(retry_delay)

    def alltalk_gradio(): # pylint: disable=too-many-statements, too-many-branches
        """
        Setup the main gradio interface for AllTalk
        """
        debug_func_entry()

        # Get the URL IP or domain name
        def get_domain_name(request: gr.Request):
            """
            Get domain name based on environment and request headers.
            Args:
                request (gr.Request): Gradio request object
            Returns:
                str: Error message if domain can't be retrieved, None otherwise
            """
            if _state['running_on_google_colab']:
                _state['my_current_url'] = _state['tunnel_url_1']
                return None

            if _state['running_in_docker']:
                _state['my_current_url'] = _state['docker_url']
                return None

            if request:
                host = request.headers.get("host", "Unknown")
                _state['my_current_url'] = host.split(":")[0]  # Split the host by ":" and take the first part
                _state['my_current_url'] = f"{_state['my_current_url']}:{config.api_def.api_port_number}"
                return None

            return "Unable to retrieve the domain name."

        with gr.Blocks(
            css=AllTalkHelpContent.custom_css,
            theme=selected_theme,
            title="AllTalk",
            analytics_enabled=False,
        ) as app: # pylint: disable=redefined-outer-name
            with gr.Row():
                gr.Markdown("## AllTalk TTS V2")
                gr.Markdown("")
                gr.Markdown("")
                dark_mode_btn = gr.Button(
                    "Light/Dark Mode", variant="primary", size="sm"
                )
                dark_mode_btn.click(
                    None,
                    None,
                    None,
                    js="""() => {
                    if (document.querySelectorAll('.dark').length) {
                        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
                        // localStorage.setItem('darkMode', 'disabled');
                    } else {
                        document.querySelector('body').classList.add('dark');
                        // localStorage.setItem('darkMode', 'enabled');
                    }
                }""",
                    show_api=False,
                )
            if config.firstrun_splash:
                with gr.Tab("AllTalk v2 Welcome page"):

                    def modify_config():
                        config.firstrun_splash = False
                        config.save()
                        return "Welcome screen is disabled."

                    gr.Markdown(
                        AllTalkHelpContent.WELCOME, elem_classes="custom-markdown"
                    )
                    with gr.Row():
                        gr.Markdown(
                            AllTalkHelpContent.WELCOME1, elem_classes="custom-markdown"
                        )
                        gr.Markdown(
                            AllTalkHelpContent.WELCOME2, elem_classes="custom-markdown"
                        )
                    update_btn = gr.Button(
                        "Click here to hide welcome screen on the next startup"
                    )
                    update_btn.click(fn=modify_config, inputs=None, outputs=None)
            with gr.Tab("Generate TTS"):
                with gr.Row():
                    gen_text = gr.Textbox(label="Text Input", lines=6)
                if _state['running_in_docker']:
                    with gr.Row():
                        with gr.Accordion(
                            "Docker IP/URL for API Address updater", open=False
                        ):
                            with gr.Row():
                                _state['docker_url'] = (
                                    f"http://localhost:{config.api_def.api_port_number}"
                                )
                                docker_upd = gr.Textbox(
                                    label="Docker IP/URL for API Address",
                                    value=_state['docker_url'],
                                    show_label=False,
                                    scale=2,
                                )
                                update_docker_btn = gr.Button(
                                    "Update Docker IP/URL API Address", scale=1
                                )
                            with gr.Accordion(
                                "HELP - 🐳 Docker IP/URL for API Addresss", open=False
                            ):
                                with gr.Row():
                                    gr.Markdown(
                                        AllTalkHelpContent.DOCKER_EXPLAINER,
                                        elem_classes="custom-markdown",
                                    )
                with gr.Row():
                    with gr.Group():
                        with gr.Row():
                            engine_choices = gr.Dropdown(
                                choices=_state['srv_engines_available'],
                                label="TTS Engine",
                                value=_state['srv_current_engine_loaded'],
                                show_label=False,
                            )
                            engine_btn = gr.Button("Swap TTS Engine")
                    with gr.Group():
                        with gr.Row():
                            model_choices_gr = gr.Dropdown(
                                choices=_state['srv_models_available'],
                                label="TTS Models",
                                value=_state['srv_current_model_loaded'],
                                interactive=True,
                                show_label=False,
                                allow_custom_value=True,
                            )
                            model_btn_gr = gr.Button("Load Different Model")
                with gr.Group():
                    at_available_voices_gr = _state['srv_current_voices']
                    rvcat_available_voices_gr = _state['srv_current_rvcvoices']
                    with gr.Row():
                        at_default_voice_gr = config.tgwui.tgwui_character_voice
                        if at_default_voice_gr not in at_available_voices_gr:
                            at_default_voice_gr = (
                                at_available_voices_gr[0]
                                if at_available_voices_gr
                                else ""
                            )
                        gen_char = gr.Dropdown(
                            choices=at_available_voices_gr,
                            label="Character Voice",
                            value=at_default_voice_gr,
                            allow_custom_value=True,
                        )
                        rvcat_default_voice_gr = config.rvc_settings.rvc_char_model_file
                        if rvcat_default_voice_gr not in rvcat_available_voices_gr:
                            rvcat_default_voice_gr = (
                                rvcat_available_voices_gr[0]
                                if rvcat_available_voices_gr
                                else ""
                            )
                        rvcgen_char = gr.Dropdown(
                            choices=rvcat_available_voices_gr,
                            label="RVC Character Voice",
                            value=rvcat_default_voice_gr,
                            allow_custom_value=True,
                        )
                        at_narrator_voice_gr = config.tgwui.tgwui_narrator_voice
                        if at_narrator_voice_gr not in at_available_voices_gr:
                            at_narrator_voice_gr = (
                                at_available_voices_gr[0]
                                if at_available_voices_gr
                                else ""
                            )
                        gen_narr = gr.Dropdown(
                            choices=at_available_voices_gr,
                            label="Narrator Voice",
                            value=at_narrator_voice_gr,
                            allow_custom_value=True,
                        )
                        rvcat_narrator_voice_gr = (
                            config.rvc_settings.rvc_narr_model_file
                        )
                        if rvcat_narrator_voice_gr not in rvcat_available_voices_gr:
                            rvcat_narrator_voice_gr = (
                                rvcat_available_voices_gr[0]
                                if rvcat_available_voices_gr
                                else ""
                            )
                        rvcgen_narr = gr.Dropdown(
                            choices=rvcat_available_voices_gr,
                            label="RVC Narrator Voice",
                            value=rvcat_narrator_voice_gr,
                            allow_custom_value=True,
                        )
                with gr.Group():
                    with gr.Row():
                        rvcat_default_pitch_gr = gr.Slider(
                            minimum=-24,
                            maximum=24,
                            step=1,
                            label="RVC Character Pitch",
                            info="Corrects the Character input TTS pitch to match the desired RVC voice output pitch.",
                            value=config.rvc_settings.pitch,
                            interactive=True,
                            visible=False,
                        )
                        rvcat_narrator_pitch_gr = gr.Slider(
                            minimum=-24,
                            maximum=24,
                            step=1,
                            label="RVC Narrator Pitch",
                            info="Corrects the Narrator input TTS pitch to match the desired RVC voice output pitch.",
                            value=config.rvc_settings.pitch,
                            interactive=True,
                            visible=False,
                        )

                        def update_visibility(char_voice, narr_voice):
                            is_visible = (
                                char_voice != "Disabled" or narr_voice != "Disabled"
                            )
                            return gr.update(visible=is_visible), gr.update(
                                visible=is_visible
                            )

                        rvcgen_char.change(
                            fn=update_visibility,
                            inputs=[rvcgen_char, rvcgen_narr],
                            outputs=[rvcat_default_pitch_gr, rvcat_narrator_pitch_gr],
                        )
                        rvcgen_narr.change(
                            fn=update_visibility,
                            inputs=[rvcgen_char, rvcgen_narr],
                            outputs=[rvcat_default_pitch_gr, rvcat_narrator_pitch_gr],
                        )

                with gr.Accordion("Advanced Engine/Model Settings", open=False):
                    with gr.Row():
                        # Get the current URL from the page
                        domain_name_output = gr.Textbox(
                            label="Domain Name", visible=False
                        )

                        def on_load(request: gr.Request):
                            domain_name = get_domain_name(request)
                            domain_name_output.value = domain_name

                        app.load(on_load, inputs=None, outputs=None)
                        if _state['srv_settings_capabilities']['streaming_capable']:
                            gen_choices = [
                                ("Standard", "false"),
                                ("Streaming (Disable Narrator)", "true"),
                            ]
                        else:
                            gen_choices = [("Standard", "false")]
                        # Continue on with the Gradio interface
                        gen_stream = gr.Dropdown(
                            choices=gen_choices, label="Generation Mode", value="false"
                        )
                        gen_lang = gr.Dropdown(
                            value=config.api_def.api_language,
                            choices=[
                                "auto",
                                "ar",
                                "zh",
                                "cs",
                                "nl",
                                "en",
                                "fr",
                                "de",
                                "hi",
                                "hu",
                                "it",
                                "ja",
                                "ko",
                                "pl",
                                "pt",
                                "ru",
                                "es",
                                "tr",
                            ],
                            label=(
                                "Languages"
                                if _state['srv_settings_capabilities']['languages_capable']
                                else "Model not multi language"
                            ),
                            interactive=_state['srv_settings_capabilities']['languages_capable'],
                            allow_custom_value=True,
                        )
                        gen_narren = gr.Dropdown(
                            choices=[
                                ("Enabled", "true"),
                                ("Disabled", "false"),
                                ("Enabled (Silent)", "silent"),
                            ],
                            label="Narrator Enabled/Disabled",
                            value=(
                                "true"
                                if config.api_def.api_narrator_enabled == "true"
                                else (
                                    "silent"
                                    if config.api_def.api_narrator_enabled == "silent"
                                    else "false"
                                )
                            ),
                            allow_custom_value=True,
                        )
                        gen_textni = gr.Dropdown(
                            choices=[
                                ("Character", "character"),
                                ("Narrator", "narrator"),
                                ("Silent", "silent"),
                            ],
                            label="Narrator Text-not-inside",
                            value=config.api_def.api_text_not_inside,
                        )
                        gen_stopcurrentgen = gr.Dropdown(
                            choices={("Stop", "true"), ("Dont stop", "false")},
                            label="Auto-Stop current generation",
                            value="true",
                        )
                    with gr.Row():
                        gen_filter = gr.Dropdown(
                            value=config.api_def.api_text_filtering,
                            label="Text filtering",
                            choices=["none", "standard", "html"],
                        )
                        gen_filetime = gr.Dropdown(
                            choices=[
                                ("Timestamp files", "true"),
                                ("Dont Timestamp (Over-write)", "false"),
                            ],
                            label="Include Timestamp",
                            value=(
                                "true"
                                if config.api_def.api_output_file_timestamp
                                else "false"
                            ),
                            allow_custom_value=True,
                        )
                        gen_autopl = gr.Dropdown(
                            choices={
                                ("Play locally", "false"),
                                ("Play remotely", "true"),
                            },
                            label="Play Locally or Remotely",
                            value="true" if config.api_def.api_autoplay else "false",
                            allow_custom_value=True,
                        )
                        gen_autoplvol = gr.Dropdown(
                            choices=[str(i / 10) for i in range(11)],
                            value=str(config.api_def.api_autoplay_volume),
                            label="Remote play volume",
                            allow_custom_value=True,
                        )
                        gen_filen = gr.Textbox(
                            value=config.api_def.api_output_file_name,
                            label="Output File Name",
                        )
                    with gr.Row():
                        gen_speed = gr.Slider(
                            minimum=0.25,
                            maximum=2.00,
                            step=0.25,
                            label="Speed",
                            value="1.00",
                            interactive=_state['srv_settings_capabilities']['generationspeed_capable'],
                        )
                        gen_pitch = gr.Slider(
                            minimum=-10,
                            maximum=10,
                            step=1,
                            label="Pitch",
                            value="1",
                            interactive=_state['srv_settings_capabilities']['pitch_capable'],
                        )
                        gen_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05,
                            label="Temperature",
                            value=0.75,
                            interactive=_state['srv_settings_capabilities']['temperature_capable'],
                        )
                        gen_repetition = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            step=1.0,
                            label="Repetition Penalty",
                            value=10,
                            interactive=_state['srv_settings_capabilities']['repetitionpenalty_capable']
                        )

                # Toggle narrator selection on Streaming select
                def update_narren_and_autopl(gen_stream):
                    debug_func_entry()
                    if gen_stream == "true":
                        return "false", "false"
                    return gen_narren.value, gen_autopl.value

                gen_stream.change(
                    update_narren_and_autopl,
                    inputs=[gen_stream],
                    outputs=[gen_narren, gen_autopl],
                )
                with gr.Row():
                    output_audio = gr.Audio(
                        show_label=False,
                        label="Generated Audio",
                        autoplay=True,
                        scale=3,
                    )
                    output_message = gr.Textbox(label="Status/Result", lines=5, scale=1)
                with gr.Row():
                    dark_mode_btn = gr.Button("Light/Dark Mode", variant="primary")
                    refresh_button = gr.Button(
                        "Refresh Server Settings", elem_id="refresh_button"
                    )
                    stop_button = gr.Button("Interupt TTS Generation")
                    submit_button = gr.Button("Generate TTS")

                model_btn_gr.click(
                    fn=change_model_loaded,
                    inputs=[engine_choices, model_choices_gr],
                    outputs=[
                        output_message,
                        gen_stream,
                        gen_char,
                        rvcgen_char,
                        gen_narr,
                        rvcgen_narr,
                        gen_speed,
                        gen_pitch,
                        gen_temperature,
                        gen_repetition,
                        gen_lang,
                        model_choices_gr,
                        engine_choices,
                    ],
                )
                engine_btn.click(
                    fn=set_engine_loaded,
                    inputs=[engine_choices],
                    outputs=[
                        output_message,
                        gen_stream,
                        gen_char,
                        rvcgen_char,
                        gen_narr,
                        rvcgen_narr,
                        gen_speed,
                        gen_pitch,
                        gen_temperature,
                        gen_repetition,
                        gen_lang,
                        model_choices_gr,
                        engine_choices,
                    ],
                )

                dark_mode_btn.click(
                    None,
                    None,
                    None,
                    js="""() => {
                            if (document.querySelectorAll('.dark').length) {
                                document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
                                // localStorage.setItem('darkMode', 'disabled');
                            } else {
                                document.querySelector('body').classList.add('dark');
                                // localStorage.setItem('darkMode', 'enabled');
                            }
                        }""",
                    show_api=False,
                )
                refresh_button.click(
                    at_update_dropdowns,
                    None,
                    [gen_stream, gen_char, rvcgen_char, gen_narr, rvcgen_narr, gen_speed, gen_pitch,
                    gen_temperature, gen_repetition, gen_lang, model_choices_gr, engine_choices],
                )
                stop_button.click(
                    stop_generate_tts, inputs=[], outputs=[output_message]
                )
                submit_button.click(
                    generate_tts,
                    inputs=[
                        gen_text,
                        gen_char,
                        rvcgen_char,
                        rvcat_default_pitch_gr,
                        gen_narr,
                        rvcgen_narr,
                        rvcat_narrator_pitch_gr,
                        gen_narren,
                        gen_textni,
                        gen_repetition,
                        gen_lang,
                        gen_filter,
                        gen_speed,
                        gen_pitch,
                        gen_autopl,
                        gen_autoplvol,
                        gen_filen,
                        gen_temperature,
                        gen_filetime,
                        gen_stream,
                        gen_stopcurrentgen,
                    ],
                    outputs=[output_audio, output_message],
                )
                if _state['running_in_docker']:
                    update_docker_btn.click(
                        fn=update_docker_address,
                        inputs=[docker_upd],
                        outputs=[output_message],
                    )
                if config.gradio_pages.Generate_Help_page:
                    with gr.Accordion("HELP - 🎯 TTS Generation Basics", open=False):
                        with gr.Row():
                            gr.Markdown(
                                AllTalkHelpContent.GENERATE_SCREEN1,
                                elem_classes="custom-markdown",
                            )
                            gr.Markdown(
                                AllTalkHelpContent.GENERATE_SCREEN2,
                                elem_classes="custom-markdown",
                            )
                    with gr.Accordion("HELP - ⚙️ Advanced TTS Features", open=False):
                        with gr.Row():
                            gr.Markdown(
                                AllTalkHelpContent.GENERATE_SCREEN3,
                                elem_classes="custom-markdown",
                            )
                            gr.Markdown(
                                AllTalkHelpContent.GENERATE_SCREEN4,
                                elem_classes="custom-markdown",
                            )

            if config.gradio_pages.Voice2RVC_page:
                with gr.Tab("Voice2RVC"):
                    with gr.Row():
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="numpy",
                            label="Record audio or Upload a spoken audio file",
                        )
                    if _state['running_in_docker']:
                        with gr.Row():
                            with gr.Accordion(
                                "Docker IP/URL for API Address updater", open=False
                            ):
                                with gr.Row():
                                    _state['docker_url'] = f"http://localhost:{config.api_def.api_port_number}"
                                    docker_upd = gr.Textbox(
                                        label="Docker IP/URL for API Address",
                                        value=_state['docker_url'],
                                        show_label=False,
                                    )
                                    update_docker_btn = gr.Button(
                                        "Update Docker IP/URL API Address"
                                    )
                                update_docker_btn.click(
                                    fn=update_docker_address,
                                    inputs=[docker_upd],
                                    outputs=[gr.Text(label="Status")],
                                )
                    with gr.Row():
                        rvc_voices_dropdown = gr.Dropdown(
                            choices=_state['srv_current_rvcvoices'],
                            label="Select RVC Voice to generate as",
                            value=_state['srv_current_rvcvoices'][0],
                            scale=1,
                        )
                        # pylint: disable=line-too-long
                        rvc_pitch_slider = gr.Slider(
                            minimum=-24,
                            maximum=24,
                            step=1,
                            label="RVC Pitch",
                            info="Depending on the pitch of your input audio, you will need to adjust this accordingly to change the pitch for the output voice. The higher the value, the higher the pitch.",
                            value=0,
                            interactive=True,
                            scale=2,
                        )
                    with gr.Row():
                        # pylint: disable=line-too-long
                        rvc_f0method = gr.Radio(
                            label="Pitch Extraction Algorithm",
                            info="Select the algorithm to be used for extracting the pitch (F0) during audio conversion. The default algorithm is rmvpe, which is generally recommended for most cases due to its balance of accuracy and performance.",
                            choices=[
                                "crepe",
                                "crepe-tiny",
                                "dio",
                                "fcpe",
                                "harvest",
                                "hybrid[rmvpe+fcpe]",
                                "pm",
                                "rmvpe",
                            ],
                            value=config.rvc_settings.f0method,
                            interactive=True,
                            scale=1,
                        )
                        submit_button = gr.Button("Submit to RVC", scale=0)
                    audio_output = gr.Audio(label="Converted Audio")

                    submit_button.click(
                        fn=voice2rvc,
                        inputs=[
                            audio_input,
                            rvc_voices_dropdown,
                            rvc_pitch_slider,
                            rvc_f0method,
                        ],
                        outputs=audio_output,
                    )
                    with gr.Accordion("HELP - 🎯 Voice2RVC Basics", open=False):
                        with gr.Row():
                            gr.Markdown(
                                AllTalkHelpContent.VOICE2RVC,
                                elem_classes="custom-markdown",
                            )
                        with gr.Row():
                            gr.Markdown(
                                AllTalkHelpContent.VOICE2RVC1,
                                elem_classes="custom-markdown",
                            )
                            gr.Markdown(
                                AllTalkHelpContent.VOICE2RVC2,
                                elem_classes="custom-markdown",
                            )

            with gr.Tab("Transcribe"):
                with gr.Row():
                    with gr.Group():
                        audio_files = gr.File(
                            file_count="multiple",
                            label="Upload Audio Files",
                            elem_classes="small-file-upload",
                        )
                        delete_after_process = gr.Checkbox(
                            label="Clean Up Temporary Audio Files After Processing",
                            value=False,
                        )
                    with gr.Row():
                        with gr.Group():
                            with gr.Row():
                                prefix_input = gr.Textbox(
                                    label="Output Prefix (optional)",
                                    placeholder="e.g., project_name, meeting_date",
                                    value="",
                                    scale=3,
                                )
                                model_choices = gr.Dropdown(
                                    choices=[
                                    "tiny",
                                    "base",
                                    "small",
                                    "medium",
                                    "turbo",
                                    "large-v3",
                                    "large-v3-turbo",
                                    ],
                                    value="turbo",
                                    label="Whisper Model Size",
                                    scale=1,
                                )
                                format_choices = gr.Dropdown(
                                    choices=["txt", "json", "srt"],
                                    value="txt",
                                    label="Output Format",
                                    scale=1,
                                )
                            with gr.Row():
                                delete_btn = gr.Button("Delete Uploaded Audio")
                                process_btn = gr.Button("Transcribe", variant="primary")
                            with gr.Row():
                                status_output = gr.Textbox(
                                    label="Processing Status", lines=2
                                )

                with gr.Row():
                    output_zip = gr.File(
                        label="Download Transcriptions (ZIP)",
                        elem_classes="small-file-upload2",
                    )

                with gr.Accordion("HELP - 🎯 Transcribe Basics", open=False):
                    with gr.Row():
                        gr.Markdown(
                            AllTalkHelpContent.TRANSCRIBE,
                            elem_classes="custom-markdown",
                        )
                    with gr.Row():
                        gr.Markdown(
                            AllTalkHelpContent.TRANSCRIBE1,
                            elem_classes="custom-markdown",
                        )
                        gr.Markdown(
                            AllTalkHelpContent.TRANSCRIBE2,
                            elem_classes="custom-markdown",
                        )

                process_btn.click(
                    fn=process_audio_files,
                    inputs=[
                        audio_files,
                        model_choices,
                        format_choices,
                        delete_after_process,
                        prefix_input,
                    ],
                    outputs=[output_zip, status_output],
                )

                delete_btn.click(
                    fn=delete_uploaded_files, inputs=[], outputs=[status_output]
                )

            with gr.Tab("Dictate"):
                state = gr.State(None)
                # Sort the dictionary by full names alphabetically
                sorted_languages = dict(
                    sorted(
                        AllTalkHelpContent.WHISPER_LANGUAGES.items(),
                        key=lambda item: item[1],
                    )
                )
                # Reverse the dictionary for lookups (Full Name -> Code)
                name_to_code = {name: code for code, name in sorted_languages.items()}

                def process_language(language_name):
                    # Convert selected language name back to its 2-digit code
                    return name_to_code[language_name]

                with gr.Row():
                    model_choices = gr.Dropdown(
                        choices=[
                            "tiny",
                            "base",
                            "small",
                            "medium",
                            "turbo",
                            "large-v3",
                            "large-v3-turbo",
                        ],
                        value="turbo",
                        label="Whisper Model",
                        scale=1,
                    )
                    language_select = gr.Dropdown(
                        choices=list(sorted_languages.values()),
                        value="English",
                        label="Language",
                        allow_custom_value=True,
                    )
                    language_select.change(
                        fn=process_language,
                        inputs=language_select,
                    )
                    export_format = gr.Dropdown(
                        choices=["txt", "srt", "json"],
                        value="txt",
                        label="Output Format",
                        scale=1,
                    )
                    prefix_input = gr.Textbox(
                        label="Output File Prefix (optional)",
                        placeholder="e.g., meeting_notes, todo_list",
                        scale=2,
                    )
                    start_btn = gr.Button("Load Model", variant="primary")
                    finish_btn = gr.Button("Finish & Unload", interactive=False)

                with gr.Accordion(
                    "Advanced Settings (Change Before Loading The Model)", open=False
                ):
                    with gr.Row():
                        # Update the dropdowns
                        translate_to_english = gr.Dropdown(
                            label="Translate to English",
                            choices={False, True},
                            value=False,
                            allow_custom_value=True,
                        )
                        source_language = gr.Dropdown(
                            choices=list(sorted_languages.values()),
                            value="English",
                            label="Source Language",
                            allow_custom_value=True,
                        )
                        source_language.change(
                            fn=process_language,
                            inputs=source_language,
                        )
                        add_timestamps = gr.Dropdown(
                            label="Add Timestamps", choices={False, True}, value=False
                        )
                        enable_diarization = gr.Dropdown(
                            label="Enable Speaker Diarization",
                            choices={False, True},
                            value=False,
                        )

                    with gr.Row():
                        enable_audio_processing = gr.Checkbox(
                            label="Enable ALL Audio Enhancements", value=True
                        )
                        bandpass_filter = gr.Checkbox(
                            label="Apply Bandpass Filter", value=True
                        )
                        noise_reduction = gr.Checkbox(
                            label="Apply Noise Reduction", value=True
                        )
                        compression = gr.Checkbox(
                            label="Apply Audio Compression", value=True
                        )

                        # At the start of interface setup
                        checkbox_state = gr.State({"processing": False})

                        def update_audio_settings(main_enable, state_dict):
                            """Enable all individual settings when main is checked"""
                            if not state_dict["processing"]:
                                state_dict["processing"] = True
                                state_dict["processing"] = False
                                # Return individual updates and state separately
                                return (
                                    gr.update(value=main_enable),  # bandpass
                                    gr.update(value=main_enable),  # noise
                                    gr.update(value=main_enable),  # compression
                                    state_dict,
                                )
                            return (
                                gr.update(),  # bandpass
                                gr.update(),  # noise
                                gr.update(),  # compression
                                state_dict,
                            )

                        def update_main_setting(bandpass, noise, compress, state_dict):
                            """Main should only be checked if all individuals are checked"""
                            if not state_dict["processing"]:
                                state_dict["processing"] = True
                                ret = gr.update(value=all([bandpass, noise, compress]))
                                state_dict["processing"] = False
                                return ret, state_dict
                            return gr.update(), state_dict

                        # In interface:
                        enable_audio_processing.change(
                            fn=update_audio_settings,
                            inputs=[enable_audio_processing, checkbox_state],
                            outputs=[
                                bandpass_filter,
                                noise_reduction,
                                compression,
                                checkbox_state,
                            ],
                        )

                        for checkbox in [bandpass_filter, noise_reduction, compression]:
                            checkbox.change(
                                fn=update_main_setting,
                                inputs=[
                                    bandpass_filter,
                                    noise_reduction,
                                    compression,
                                    checkbox_state,
                                ],
                                outputs=[enable_audio_processing, checkbox_state],
                            )

                    with gr.Row():
                        silence_threshold = gr.Slider(
                            minimum=0.001,
                            maximum=0.02,
                            value=0.008,
                            label="Silence Threshold",
                        )
                        bandpass_low = gr.Slider(
                            minimum=50,
                            maximum=200,
                            value=75,
                            label="Bandpass Low Freq (Hz)",
                        )
                        bandpass_high = gr.Slider(
                            minimum=2000,
                            maximum=8000,
                            value=3800,
                            label="Bandpass High Freq (Hz)",
                        )

                with gr.Row():
                    # Audio input with waveform visualization
                    waveform_opts = gr.WaveformOptions(
                        sample_rate=16000,  # Set to Whisper's expected rate
                        show_recording_waveform=True,
                        show_controls=False,
                        waveform_color="#1f77b4",
                        waveform_progress_color="#2ecc71",
                    )

                    # Audio input with configured waveform
                    dictate_audio = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        streaming=True,
                        label="Click `Record` to start dictation and `Stop` to pause or stop dictation",
                        show_label=True,
                        interactive=False,
                        waveform_options=waveform_opts,
                    )
                with gr.Accordion("Show Audio Levels Graph", open=False, elem_classes="fixed-accordion"):
                    audio_plot = gr.Plot(
                        label="Audio Levels",
                        show_label=True,
                        container=True,
                        elem_classes="fixed-plot",
                    )
                with gr.Row():
                    text_output = gr.Textbox(
                        label="Live Transcription",
                        lines=10,
                        placeholder="Transcription will appear here as you speak...",
                        show_copy_button=True,
                    )

                # Dictatation Button click handlers
                start_btn.click(
                    fn=start_new_dictation,
                    inputs=[
                        model_choices,
                        prefix_input,
                        language_select,
                        source_language,
                        translate_to_english,
                        export_format,
                        add_timestamps,
                        enable_diarization,
                        enable_audio_processing,
                        bandpass_filter,
                        noise_reduction,
                        compression,
                        silence_threshold,
                        bandpass_low,
                        bandpass_high,
                    ],
                    outputs=[state, text_output, dictate_audio, start_btn, finish_btn],
                )

                finish_btn.click(
                    fn=finish_dictation,
                    inputs=[state, dictate_audio, text_output, audio_plot],  # Pass the actual components
                    outputs=[
                        state,               # Update the state
                        text_output,         # Update the transcription text output
                        start_btn,           # Update the start button
                        finish_btn,          # Update the finish button
                    ],
                )


                def on_start_recording(state):
                    if state and state.get("is_active"):
                        print_message("Recording started", "debug_transcribe")
                        state["is_recording"] = True
                    return state

                def on_stop_recording(state):
                    if state and state.get("is_active"):
                        print_message("Recording stopped", "debug_transcribe")
                        state["is_recording"] = False
                    return state


                dictate_audio.start_recording(
                    fn=on_start_recording, inputs=[state], outputs=[state]
                ).success(fn=None, js="() => {console.log('Recording started');}")

                dictate_audio.stop_recording(
                    fn=on_stop_recording, inputs=[state], outputs=[state]
                ).success(fn=None, js="() => {console.log('Recording stopped');}")

                dictate_audio.stream(
                    fn=process_audio,
                    inputs=[dictate_audio, state],
                    outputs=[state, text_output, audio_plot],
                    show_progress=False,
                    concurrency_limit=1,
                )

                with gr.Accordion("HELP - 🎯 Dictate Basics", open=False):
                    with gr.Row():
                        gr.Markdown(
                            AllTalkHelpContent.WHISPER_HELP,
                            elem_classes="custom-markdown",
                        )
                    with gr.Row():
                        gr.Markdown(
                            AllTalkHelpContent.WHISPER_HELP1,
                            elem_classes="custom-markdown",
                        )
                        gr.Markdown(
                            AllTalkHelpContent.WHISPER_HELP2,
                            elem_classes="custom-markdown",
                        )

            if config.gradio_pages.TTS_Generator_page:
                with gr.Tab("TTS Generator"):
                    # pylint: disable=line-too-long
                    gr.Markdown(
                        """### TTS Generator for long audio generation tasks. [Click here for access](http://127.0.0.1:7851/static/tts_generator/tts_generator.html)"""
                    )
                    with gr.Accordion("HELP -  🎯 TTS Generator Basics", open=False):
                        with gr.Row():
                            gr.Markdown(
                                AllTalkHelpContent.TTS_GENERATOR,
                                elem_classes="custom-markdown",
                            )
                        with gr.Row():
                            gr.Markdown(
                                AllTalkHelpContent.TTS_GENERATOR1,
                                elem_classes="custom-markdown",
                            )
                            gr.Markdown(
                                AllTalkHelpContent.TTS_GENERATOR2,
                                elem_classes="custom-markdown",
                            )

            with gr.Tab("Global Settings"):
                with gr.Tab("AllTalk Settings"):
                    with gr.Row():
                        delete_output_wavs = gr.Dropdown(
                            value=config.delete_output_wavs,
                            label="Del WAV's older than",
                            choices=[
                                "Disabled",
                                "1 Day",
                                "2 Days",
                                "3 Days",
                                "4 Days",
                                "5 Days",
                                "6 Days",
                                "7 Days",
                                "14 Days",
                                "21 Days",
                                "28 Days",
                            ],
                        )
                        api_port_number = gr.Number(
                            value=config.api_def.api_port_number,
                            label="API Port Number",
                            precision=0,
                        )
                        gradio_port_number = gr.Number(
                            value=config.gradio_port_number,
                            label="Gradio Port Number",
                            precision=0,
                        )
                        settings_output_folder = gr.Textbox(
                            value=config.output_folder,
                            label=f"Output Folder name (sub {config.branding})",
                        )
                    with gr.Row():
                        transcode_audio_format = gr.Dropdown(
                            choices={
                                "Disabled": "disabled",
                                "aac": "aac",
                                "flac": "flac",
                                "mp3": "mp3",
                                "opus": "opus",
                                "wav": "wav",
                            },
                            label="Audio Transcoding",
                            value=config.transcode_audio_format,
                        )
                        with gr.Row():
                            themes_select = gr.Dropdown(
                                loadThemes.get_list(),
                                value=loadThemes.read_json(),
                                label="Gradio Theme Selection",
                                visible=True,
                            )

                            def update_theme_selection(theme_name):
                                updtheme_config = AlltalkConfig.get_instance()
                                updtheme_config.theme.clazz = theme_name
                                updtheme_config.save()  # Save the updated configuration
                                return (
                                    f"Theme '{theme_name}' has been selected and saved."
                                )

                            themes_select.change(
                                fn=update_theme_selection,
                                inputs=[themes_select],
                                outputs=[gr.Textbox(label="Gradio Selection Result")],
                            )
                    with gr.Row():
                        with gr.Column():
                            gr_debug_tts = gr.CheckboxGroup(
                                choices=debugging_choices,
                                label="Debugging Options list",
                                value=default_values,
                            )
                        with gr.Column():
                            # pylint: disable=line-too-long
                            gradio_interface = gr.Dropdown(
                                choices={"Enabled": "true", "Disabled": "false"},
                                label="Gradio Interface",
                                value=(
                                    "Enabled" if config.gradio_interface else "Disabled"
                                ),
                                info="**WARNING**: This will disable the AllTalk Gradio interface from loading. To re-enable the interface, go to the API address in a web browser and enable it there. http://127.0.0.1:7851/",
                                allow_custom_value=True,
                            )
                    with gr.Group():
                        gr.Markdown("### Disable Gradio Interface Tabs")
                        with gr.Row():
                            generate_help_page = gr.Checkbox(
                                label="Generate Help",
                                value=config.gradio_pages.Generate_Help_page,
                            )
                            voice2rvc_page = gr.Checkbox(
                                label="Voice2RVC",
                                value=config.gradio_pages.Voice2RVC_page,
                            )
                            tts_generator_page = gr.Checkbox(
                                label="TTS Generator",
                                value=config.gradio_pages.TTS_Generator_page,
                            )
                            tts_engines_settings_page = gr.Checkbox(
                                label="TTS Engines Settings",
                                value=config.gradio_pages.TTS_Engines_Settings_page,
                            )
                            alltalk_documentation_page = gr.Checkbox(
                                label="AllTalk Documentation",
                                value=config.gradio_pages.alltalk_documentation_page,
                            )
                            api_documentation_page = gr.Checkbox(
                                label="API Documentation",
                                value=config.gradio_pages.api_documentation_page,
                            )

                    with gr.Row():
                        submit_button = gr.Button("Update Settings")
                        output_message = gr.Textbox(
                            label="Output Message", interactive=False, show_label=False
                        )

                    with gr.Accordion("⚙️ HELP - AllTalk Settings Page", open=False):
                        with gr.Row():
                            gr.Markdown(
                                AllTalkHelpContent.ALLTALK_SETTINGS_PAGE1,
                                elem_classes="custom-markdown",
                            )
                            gr.Markdown(
                                AllTalkHelpContent.ALLTALK_SETTINGS_PAGE2,
                                elem_classes="custom-markdown",
                            )

                    with gr.Accordion("🔍 HELP - AllTalk Debug Settings", open=False):
                        with gr.Row():
                            gr.Markdown(
                                AllTalkHelpContent.DEBUG_HELP1,
                                elem_classes="custom-markdown",
                            )
                            gr.Markdown(
                                AllTalkHelpContent.DEBUG_HELP2,
                                elem_classes="custom-markdown",
                            )

                    # Update the function to include these new settings
                    submit_button.click(
                        update_settings_at,
                        inputs=[
                            delete_output_wavs,
                            gradio_interface,
                            gradio_port_number,
                            settings_output_folder,
                            api_port_number,
                            gr_debug_tts,
                            transcode_audio_format,
                            generate_help_page,
                            voice2rvc_page,
                            tts_generator_page,
                            tts_engines_settings_page,
                            alltalk_documentation_page,
                            api_documentation_page,
                        ],
                        outputs=output_message,
                    )
                with gr.Tab("AllTalk API Defaults"):
                    with gr.Row():
                        # API Version Settings
                        with gr.Column(scale=1):
                            with gr.Group():
                                with gr.Row():
                                    api_use_legacy_api = gr.Dropdown(
                                        choices=[
                                            "AllTalk v2 API",
                                            "AllTalk v1 API (Legacy)",
                                        ],
                                        label="API Version",
                                        scale=1,
                                        allow_custom_value=True,
                                        value=config.api_def.api_use_legacy_api,
                                    )
                                    api_legacy_ip_address = gr.Textbox(
                                        value=config.api_def.api_legacy_ip_address,
                                        scale=1,
                                        label="Legacy API IP Address",
                                    )
                        # Request Processing
                        with gr.Column(scale=2):
                            with gr.Group():
                                with gr.Row():
                                    api_length_stripping = gr.Slider(
                                        minimum=1,
                                        maximum=20,
                                        step=1,
                                        value=int(config.api_def.api_length_stripping),
                                        scale=2,
                                        label="Minimum Sentence Length",
                                    )
                                    api_max_characters = gr.Slider(
                                        minimum=50,
                                        maximum=10000,
                                        step=50,
                                        value=int(config.api_def.api_max_characters),
                                        scale=2,
                                        label="Maximum Request Characters",
                                    )
                                    api_text_filtering = gr.Dropdown(
                                        value=config.api_def.api_text_filtering,
                                        label="Text Filtering Mode",
                                        scale=2,
                                        choices=["none", "standard", "html"],
                                    )
                                    api_language = gr.Dropdown(
                                        value=config.api_def.api_language,
                                        label="Default Language",
                                        scale=2,
                                        choices=[
                                            "ar",
                                            "zh",
                                            "cs",
                                            "nl",
                                            "en",
                                            "fr",
                                            "de",
                                            "hi",
                                            "hu",
                                            "it",
                                            "ja",
                                            "ko",
                                            "pl",
                                            "pt",
                                            "ru",
                                            "es",
                                            "tr",
                                        ],
                                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group():
                                with gr.Row():
                                    api_narrator_enabled = gr.Dropdown(
                                        choices=[
                                            ("Enabled", "true"),
                                            ("Disabled", "false"),
                                            ("Enabled (Silent)", "silent"),
                                        ],
                                        label="Narrator Mode",
                                        allow_custom_value=True,
                                        value=(
                                            "true"
                                            if config.api_def.api_narrator_enabled
                                            == "true"
                                            else (
                                                "silent"
                                                if config.api_def.api_narrator_enabled
                                                == "silent"
                                                else "false"
                                            )
                                        ),
                                    )
                                    api_text_not_inside = gr.Dropdown(
                                        choices=["character", "narrator", "silent"],
                                        label="Text-Not-Inside Handling",
                                        allow_custom_value=True,
                                        value=config.api_def.api_text_not_inside,
                                    )
                        with gr.Column(scale=2):
                            with gr.Group():
                                with gr.Row():
                                    api_output_file_name = gr.Textbox(
                                        value=config.api_def.api_output_file_name,
                                        label="Default Filename",
                                    )
                                    api_output_file_timestamp = gr.Dropdown(
                                        choices=[
                                            "Timestamp files",
                                            "Dont Timestamp (Over-write)",
                                        ],
                                        label="File Timestamping",
                                        allow_custom_value=True,
                                        value=(
                                            "Timestamp files"
                                            if config.api_def.api_output_file_timestamp
                                            else "Dont Timestamp (Over-write)"
                                        ),
                                    )
                                    api_autoplay = gr.Dropdown(
                                        choices=["Play locally", "Play remotely"],
                                        label="Playback Location",
                                        allow_custom_value=True,
                                        value=(
                                            "Play remotely"
                                            if config.api_def.api_autoplay
                                            else "Play locally"
                                        ),
                                    )
                                    api_autoplay_volume = gr.Slider(
                                        minimum=0.1,
                                        maximum=0.9,
                                        step=0.1,
                                        label="Remote Playback Volume",
                                        value=float(config.api_def.api_autoplay_volume),
                                    )
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Group():
                                with gr.Row():
                                    api_allowed_filter = gr.Textbox(
                                        value=config.api_def.api_allowed_filter,
                                        label="Allowed Characters & Unicode Ranges",
                                        lines=3,
                                    )
                        with gr.Column(scale=1):
                            with gr.Group():
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        output_message = gr.Textbox(
                                            label="Status", interactive=False
                                        )
                                        submit_button = gr.Button("Update Settings")
                    # Help Accordions
                    with gr.Accordion("HELP - 🎯 Quick Start Guide", open=False):
                        gr.Markdown(
                            AllTalkHelpContent.API_DEFAULTS1,
                            elem_classes="custom-markdown",
                        )
                    with gr.Accordion(
                        "HELP - 🔄 Version & Compatibility Settings (API v1/v2, Legacy support)",
                        open=False,
                    ):
                        gr.Markdown(
                            AllTalkHelpContent.API_DEFAULTS2,
                            elem_classes="custom-markdown",
                        )
                    with gr.Accordion(
                        "HELP - 🎛️ Request Configuration (Character limits, language, file names)",
                        open=False,
                    ):
                        gr.Markdown(
                            AllTalkHelpContent.API_DEFAULTS3,
                            elem_classes="custom-markdown",
                        )
                    with gr.Accordion(
                        "HELP - 📝 Text Processing & Filtering (Filtering modes, character sets, language)",
                        open=False,
                    ):
                        gr.Markdown(
                            AllTalkHelpContent.API_DEFAULTS4,
                            elem_classes="custom-markdown",
                        )
                    with gr.Accordion(
                        "HELP - 🗣️ Narrator & Playback (Narrator modes, text handling, playback options",
                        open=False,
                    ):
                        gr.Markdown(
                            AllTalkHelpContent.API_DEFAULTS5,
                            elem_classes="custom-markdown",
                        )
                    with gr.Accordion(
                        "HELP - 📚 API Allowed Text Filtering/Passthrough", open=False
                    ):
                        gr.Markdown(
                            AllTalkHelpContent.API_DEFAULTS6,
                            elem_classes="custom-markdown",
                        )
                    with gr.Accordion(
                        "HELP - ❗ Troubleshooting & Best Practices", open=False
                    ):
                        gr.Markdown(
                            AllTalkHelpContent.API_DEFAULTS7,
                            elem_classes="custom-markdown",
                        )

                    submit_button.click(
                        update_settings_api,
                        inputs=[
                            api_length_stripping,
                            api_legacy_ip_address,
                            api_allowed_filter,
                            api_max_characters,
                            api_use_legacy_api,
                            api_text_filtering,
                            api_narrator_enabled,
                            api_text_not_inside,
                            api_language,
                            api_output_file_name,
                            api_output_file_timestamp,
                            api_autoplay,
                            api_autoplay_volume,
                        ],
                        outputs=output_message,
                    )

                    def rvc_update_dropdowns():
                        """Update RVC dropdowns with current values from state"""
                        debug_func_entry()

                        # Update state with current settings
                        get_alltalk_settings()

                        # Get values from state
                        current_voices = _state['srv_current_rvcvoices']
                        current_char = _state['srv_rvc_character_voice']
                        current_narr = _state['srv_rvc_narrator_voice']

                        # Handle default voices if they're not in available voices
                        if current_char not in current_voices:
                            current_char = current_voices[0] if current_voices else ""
                        if current_narr not in current_voices:
                            current_narr = current_voices[0] if current_voices else ""

                        return (
                            gr.Dropdown(choices=current_voices, value=current_char, interactive=True),
                            gr.Dropdown(choices=current_voices, value=current_narr, interactive=True)
                        )

                def gr_update_rvc_settings(
                    rvc_enabled,
                    rvc_char_model_file,
                    rvc_narr_model_file,
                    split_audio,
                    autotune,
                    pitch,
                    filter_radius,
                    index_rate,
                    rms_mix_rate,
                    protect,
                    hop_length,
                    f0method,
                    embedder_model,
                    training_data_size,
                ):
                    debug_func_entry()
                    progress = gr.Progress(track_tqdm=True)
                    return update_rvc_settings(
                        rvc_enabled,
                        rvc_char_model_file,
                        rvc_narr_model_file,
                        split_audio,
                        autotune,
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                        hop_length,
                        f0method,
                        embedder_model,
                        training_data_size,
                        progress,
                    )

                with gr.Tab("RVC Settings"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Row():
                                rvc_model_file_available = _state['srv_current_rvcvoices']
                                rvc_char_model_file_default = (
                                    config.rvc_settings.rvc_char_model_file
                                )
                                if (
                                    rvc_char_model_file_default
                                    not in rvc_model_file_available
                                ):
                                    rvc_char_model_file_default = (
                                        rvc_model_file_available[0]
                                        if rvc_model_file_available
                                        else ""
                                    )
                                rvc_char_model_file_gr = gr.Dropdown(
                                    choices=rvc_model_file_available,
                                    label="Default Character Voice Model",
                                    info="Select the Character voice model used for conversion.",
                                    value=rvc_char_model_file_default,
                                    allow_custom_value=True,
                                )
                                rvc_narr_model_file_default = (
                                    config.rvc_settings.rvc_narr_model_file
                                )
                                if (
                                    rvc_narr_model_file_default
                                    not in rvc_model_file_available
                                ):
                                    rvc_narr_model_file_default = (
                                        rvc_model_file_available[0]
                                        if rvc_model_file_available
                                        else ""
                                    )
                                rvc_narr_model_file_gr = gr.Dropdown(
                                    choices=rvc_model_file_available,
                                    label="Default Narrator Voice Model",
                                    info="Select the Narrator voice model used for conversion.",
                                    value=rvc_narr_model_file_default,
                                    allow_custom_value=True,
                                )
                                rvc_refresh_button = gr.Button("Refresh Model Choices")
                        with gr.Column(scale=0):
                            # pylint: disable=line-too-long
                            rvc_enabled = gr.Checkbox(
                                label="Enable RVC",
                                info="RVC (Real-Time Voice Cloning) enhances TTS by replicating voice characteristics for characters or narrators, adding depth to synthesized speech.",
                                value=config.rvc_settings.rvc_enabled,
                                interactive=True,
                            )
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            # pylint: disable=line-too-long
                            pitch = gr.Slider(
                                minimum=-24,
                                maximum=24,
                                step=1,
                                label="Pitch",
                                info="Set the pitch of the audio, the higher the value, the higher the pitch.",
                                value=config.rvc_settings.pitch,
                                interactive=True,
                            )
                        with gr.Column():
                            # pylint: disable=line-too-long
                            hop_length = gr.Slider(
                                minimum=1,
                                maximum=512,
                                step=1,
                                label="Hop Length",
                                info="Smaller hop lengths require more time for inference but tend to yield higher pitch accuracy.",
                                value=config.rvc_settings.hop_length,
                                interactive=True,
                            )
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            # pylint: disable=line-too-long
                            training_data_size = gr.Slider(
                                minimum=10000,
                                maximum=100000,
                                step=5000,
                                label="Training Data Size",
                                info="Determines the number of training data points used to train the FAISS index. Increasing the size may improve the quality but can also increase computation time.",
                                value=config.rvc_settings.training_data_size,
                                interactive=True,
                            )
                        with gr.Column():
                            # pylint: disable=line-too-long
                            index_rate = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="Index Influence Ratio",
                                info="Sets the influence exerted by the index file on the final output. A higher value increases the impact of the index, potentially enhancing detail but also increasing the risk of artifacts.",
                                value=config.rvc_settings.index_rate,
                                interactive=True,
                            )
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            # pylint: disable=line-too-long
                            rms_mix_rate = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="Volume Envelope",
                                info="Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed.",
                                value=config.rvc_settings.rms_mix_rate,
                                interactive=True,
                            )
                        with gr.Column():
                            # pylint: disable=line-too-long
                            protect = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label="Protect Voiceless Consonants/Breath sounds",
                                info="Prevents sound artifacts. Higher values (up to 0.5) provide stronger protection but may affect indexing.",
                                value=config.rvc_settings.protect,
                                interactive=True,
                            )
                        with gr.Column():
                            # pylint: disable=line-too-long
                            filter_radius = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label="Filter Radius",
                                info="If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration.",
                                value=config.rvc_settings.filter_radius,
                                step=1,
                                interactive=True,
                            )
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Row():
                                    embedder_model = gr.Radio(
                                        label="Embedder Model",
                                        info="Model used for learning speaker embedding.",
                                        choices=["hubert", "contentvec"],
                                        value=config.rvc_settings.embedder_model,
                                        interactive=True,
                                    )
                                with gr.Row():
                                    # pylint: disable=line-too-long
                                    split_audio = gr.Checkbox(
                                        label="Split Audio",
                                        info="Split the audio into chunks for inference to obtain better results in some cases.",
                                        value=config.rvc_settings.split_audio,
                                        interactive=True,
                                    )
                                    # pylint: disable=line-too-long
                                    autotune = gr.Checkbox(
                                        label="Autotune",
                                        info="Apply a soft autotune to your inferences, recommended for singing conversions.",
                                        value=config.rvc_settings.autotune,
                                        interactive=True,
                                    )
                        with gr.Column():
                            # pylint: disable=line-too-long
                            f0method = gr.Radio(
                                label="Pitch Extraction Algorithm",
                                info="Select the algorithm to be used for extracting the pitch (F0) during audio conversion. The default algorithm is rmvpe, which is generally recommended for most cases due to its balance of accuracy and performance.",
                                choices=[
                                    "crepe",
                                    "crepe-tiny",
                                    "dio",
                                    "fcpe",
                                    "harvest",
                                    "hybrid[rmvpe+fcpe]",
                                    "pm",
                                    "rmvpe",
                                ],
                                value=config.rvc_settings.f0method,
                                interactive=True,
                            )
                    with gr.Row():
                        update_button = gr.Button("Update RVC Settings")
                        update_output = gr.Textbox(
                            label="Update Status", show_label=False
                        )
                    rvc_refresh_button.click(
                        rvc_update_dropdowns,
                        None,
                        [rvc_char_model_file_gr, rvc_narr_model_file_gr],
                    )
                    update_button.click(
                        fn=gr_update_rvc_settings,
                        inputs=[
                            rvc_enabled,
                            rvc_char_model_file_gr,
                            rvc_narr_model_file_gr,
                            split_audio,
                            autotune,
                            pitch,
                            filter_radius,
                            index_rate,
                            rms_mix_rate,
                            protect,
                            hop_length,
                            f0method,
                            embedder_model,
                            training_data_size,
                        ],
                        outputs=[update_output],
                    )

                    with gr.Accordion(
                        "🗣️ HELP - RVC Settings Page",
                        open=False,
                        elem_classes=["gr-accordion"],
                    ):
                        with gr.Row():
                            gr.Markdown(
                                AllTalkHelpContent.RVC_PAGE1,
                                elem_classes="custom-markdown",
                            )
                            gr.Markdown(
                                AllTalkHelpContent.RVC_PAGE2,
                                elem_classes="custom-markdown",
                            )

                with gr.Tab("SSL Proxy Server"):
                    _proxy_interface = create_proxy_interface(_state['proxy_manager'])
                    with gr.Accordion("HELP - 🎯 Proxy Quick Start Guide", open=False):
                        gr.Markdown(
                            AllTalkHelpContent.PROXY, elem_classes="custom-markdown"
                        )
                        with gr.Row():
                            gr.Markdown(
                                AllTalkHelpContent.PROXY1,
                                elem_classes="custom-markdown",
                            )
                            gr.Markdown(
                                AllTalkHelpContent.PROXY2,
                                elem_classes="custom-markdown",
                            )   

                with gr.Tab("Text-generation-webui Settings"):
                    with gr.Row():
                        activate = gr.Dropdown(
                            choices={"Enabled": "true", "Disabled": "false"},
                            label="Activate TTS",
                            value=(
                                "Enabled"
                                if config.tgwui.tgwui_activate_tts
                                else "Disabled"
                            ),
                            allow_custom_value=True,
                        )
                        autoplay = gr.Dropdown(
                            choices={"Enabled": "true", "Disabled": "false"},
                            label="Autoplay TTS",
                            value=(
                                "Enabled"
                                if config.tgwui.tgwui_autoplay_tts
                                else "Disabled"
                            ),
                            allow_custom_value=True,
                        )
                        show_text = gr.Dropdown(
                            choices={"Enabled": "true", "Disabled": "false"},
                            label="Show Text",
                            value=(
                                "Enabled"
                                if config.tgwui.tgwui_show_text
                                else "Disabled"
                            ),
                            allow_custom_value=True,
                        )
                        narrator_enabled = gr.Dropdown(
                            choices=[
                                ("Enabled", "true"),
                                ("Disabled", "false"),
                                ("Enabled (Silent)", "silent"),
                            ],
                            label="Narrator enabled",
                            value=(
                                "true"
                                if config.tgwui.tgwui_narrator_enabled == "true"
                                else (
                                    "silent"
                                    if config.tgwui.tgwui_narrator_enabled == "silent"
                                    else "false"
                                )
                            ),
                            allow_custom_value=True,
                        )
                        language = gr.Dropdown(
                            value=config.tgwui.tgwui_language,
                            label="Default Language",
                            choices=_state['gradio_languages_list'],
                            allow_custom_value=True,
                        )
                    with gr.Row():
                        submit_button = gr.Button("Update Settings")
                        output_message = gr.Textbox(
                            label="Output Message", interactive=False, show_label=False
                        )

                    submit_button.click(
                        update_settings_tgwui,
                        inputs=[
                            activate,
                            autoplay,
                            show_text,
                            language,
                            narrator_enabled,
                        ],
                        outputs=output_message,
                    )

                disk_space_page = get_disk_interface()
                disk_space_page()                           

            if config.gradio_pages.TTS_Engines_Settings_page:
                with gr.Tab("TTS Engines Settings"):
                    with gr.Tabs():
                        for engine_name in _state.get('srv_engines_available', []):
                            if base_package:
                                module_name = (
                                    f"{base_package}.system.tts_engines."
                                    f"{engine_name}."
                                    f"{engine_name}_settings_page"
                                )
                            else:
                                module_name = f"system.tts_engines.{engine_name}.{engine_name}_settings_page"

                            module = dynamic_import(module_name, base_package)
                            if module:
                                with gr.Tab(f"{engine_name.capitalize()} TTS"):
                                    gr.Markdown(f"### &nbsp;&nbsp;{engine_name.capitalize()} TTS")
                                    getattr(
                                        module,
                                        f"{engine_name}_at_gradio_settings_page"
                                    )(
                                        globals()[f"{engine_name}_model_config_data"]
                                    )

            if config.gradio_pages.alltalk_documentation_page:
                with gr.Tab("Documentation"):
                    with gr.Accordion("📖 HELP - Narrator Function", open=False):
                        gr.Markdown(
                            AllTalkHelpContent.NARRATOR, elem_classes="custom-markdown"
                        )
                        with gr.Row():
                            gr.Markdown(
                                AllTalkHelpContent.NARRATOR1,
                                elem_classes="custom-markdown",
                            )
                            gr.Markdown(
                                AllTalkHelpContent.NARRATOR2,
                                elem_classes="custom-markdown",
                            )
                    with gr.Accordion(
                        "🔌 HELP - API Standard TTS Generation Endpoints", open=False
                    ):
                        gr.Markdown(
                            AllTalkHelpContent.API_STANDARD,
                            elem_classes="custom-markdown",
                        )
                    with gr.Accordion(
                        "🔌 HELP - API Streaming TTS Generation Endpoints", open=False
                    ):
                        gr.Markdown(
                            AllTalkHelpContent.API_STREAMING,
                            elem_classes="custom-markdown",
                        )
                    with gr.Accordion(
                        "🔌 HELP - API Server Control Endpoints", open=False
                    ):
                        gr.Markdown(
                            AllTalkHelpContent.API_CONTROL,
                            elem_classes="custom-markdown",
                        )
                    with gr.Accordion(
                        "🔌 HELP - API Server Status Endpoints", open=False
                    ):
                        gr.Markdown(
                            AllTalkHelpContent.API_STATUS,
                            elem_classes="custom-markdown",
                        )
                    with gr.Accordion(
                        "🔌 HELP - API OpenAI V1 Speech Endpoints", open=False
                    ):
                        gr.Markdown(
                            AllTalkHelpContent.API_OPENAPI,
                            elem_classes="custom-markdown",
                        )

            with gr.Tab("About this project"):
                with gr.Row():
                    gr.Markdown(
                        AllTalkHelpContent.WELCOME1, elem_classes="custom-markdown"
                    )
                    gr.Markdown(
                        AllTalkHelpContent.WELCOME2, elem_classes="custom-markdown"
                    )

        return app

    # pylint: disable=redefined-outer-name
    if __name__ == "__main__":
        app = alltalk_gradio().queue()
        app.launch(
            server_name="0.0.0.0",
            server_port=config.gradio_port_number,
            prevent_thread_lock=True,
            quiet=True,
        )

    if not running_in_standalone:
        app = alltalk_gradio().queue()
        app.launch(
            server_name="0.0.0.0",
            server_port=config.gradio_port_number,
            prevent_thread_lock=True,
            quiet=True,
        )

#########################################
# START-UP # Final Splash before Gradio #
#########################################
print_message("Please use \033[91mCtrl+C\033[0m when exiting AllTalk otherwise a")
print_message("subprocess may continue running in the background.")
print_message("")
print_message("Server Ready")

###############################################################################################
# START-UP # Loop to keep the script from exiting out if its being run as a standalone script #
###############################################################################################
if running_in_standalone:
    while True:
        try:
            time.sleep(1)  # Add a small delay to avoid high CPU usage
        except KeyboardInterrupt:
            break  # Allow graceful exit on Ctrl+C
