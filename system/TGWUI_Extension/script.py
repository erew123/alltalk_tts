import re
import sys
import json
import time
import inspect
import random
import logging
import requests
import threading
import gradio as gr
from pathlib import Path
from modules import chat, shared, ui_chat
from modules.utils import gradio
from requests.exceptions import RequestException, ConnectionError

# Store the current disable level
current_disable_level = logging.getLogger().manager.disable
this_dir = Path(__file__).parent.resolve()

class TGWUIModeManager:
    def __init__(self):
        self.debug_tgwui = False  # General Debugging setting. Default is `False`
        self.debug_func = False  # Print function name as its used. Default is `False`
        self.is_local = self._detect_mode()
        self._setup_config()
        self._setup_server_connection()

    def _detect_mode(self):
        """Detect if we're running as part of AllTalk installation"""
        try:
            current_dir = Path(__file__).parent
            alltalk_root = current_dir.parent.parent
            
            # Just check if we're in the right place, don't import
            required_files = [
                alltalk_root / "confignew.json",
                alltalk_root / "script.py"
            ]
            required_folders = [
                alltalk_root / "system",
                alltalk_root / "models",
                alltalk_root / "outputs",
                alltalk_root / "voices"
            ]

            files_exist = all(f.exists() for f in required_files)
            folders_exist = all(f.is_dir() for f in required_folders)

            return files_exist and folders_exist

        except (ImportError, FileNotFoundError) as e:
            # Don't try to use mode_manager here since it doesn't exist yet
            print(f"Error detecting mode: {str(e)}")
            return False
            
    def _setup_config(self):
        """Load appropriate configuration based on mode"""
        if self.is_local:
            # We're in TGWUI mode, use default settings
            self.branding = "AllTalk "
            self.server = {
                "protocol": "http://",
                "address": "127.0.0.1:7851",  # Default port
                "timeout": 5
            }
            # Set complete default config structure
            self.config = {
                "api_def": {
                    "api_port_number": 7851,
                    "api_use_legacy_api": False
                },
                "tgwui": {
                    "tgwui_activate_tts": True,
                    "tgwui_autoplay_tts": True,
                    "tgwui_narrator_enabled": "false",
                    "tgwui_non_quoted_text_is": "narrator",
                    "tgwui_deepspeed_enabled": False,
                    "tgwui_language": "English",
                    "tgwui_lowvram_enabled": False,
                    "tgwui_pitch_set": 0,
                    "tgwui_temperature_set": 0.75,
                    "tgwui_repetitionpenalty_set": 10,
                    "tgwui_generationspeed_set": 1,
                    "tgwui_narrator_voice": "Please Refresh Settings",
                    "tgwui_show_text": True,
                    "tgwui_character_voice": "Please Refresh Settings",
                    "tgwui_rvc_char_voice": "Disabled",
                    "tgwui_rvc_char_pitch": 0,
                    "tgwui_rvc_narr_voice": "Disabled",
                    "tgwui_rvc_narr_pitch": 0
                },
                "remote_connection": {  # Add this section for local mode too
                    "use_legacy_api": False,
                    "server_protocol": "http://",
                    "server_address": "127.0.0.1:7851",
                    "connection_timeout": 5
                }
            }
        else:
            config_path = Path(__file__).parent / "tgwui_remote_config.json"
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading remote config ({str(e)}), using defaults")
                self.config = {
                    "branding": "AllTalk ",
                    "tgwui": {
                        "tgwui_activate_tts": True,
                        "tgwui_autoplay_tts": False,
                        "tgwui_show_text": True,
                        "tgwui_narrator_enabled": "false",
                        "tgwui_non_quoted_text_is": "narrator",
                        "tgwui_deepspeed_enabled": False,
                        "tgwui_language": "English",
                        "tgwui_lowvram_enabled": False,
                        "tgwui_pitch_set": 0,
                        "tgwui_temperature_set": 0.75,
                        "tgwui_repetitionpenalty_set": 10,
                        "tgwui_generationspeed_set": 1.0,
                        "tgwui_narrator_voice": "Please Refresh Settings",
                        "tgwui_character_voice": "Please Refresh Settings",
                        "tgwui_rvc_char_voice": "Disabled",
                        "tgwui_rvc_char_pitch": 0,
                        "tgwui_rvc_narr_voice": "Disabled",
                        "tgwui_rvc_narr_pitch": 0                        
                    },
                    "remote_connection": {
                        "server_protocol": "http://",
                        "server_address": "127.0.0.1:7851",
                        "connection_timeout": 5,
                        "use_legacy_api": False
                    }
                }

            self.branding = self.config['branding']
            self.server = {
                "protocol": self.config['remote_connection']['server_protocol'],
                "address": self.config['remote_connection']['server_address'],
                "timeout": self.config['remote_connection']['connection_timeout']
            }        

    def _setup_server_connection(self):
        """Initialize API connection settings"""
        self.server_url = f"{self.server['protocol']}{self.server['address']}"
        
    def get_api_url(self, endpoint):
        """Get full API URL for a given endpoint"""
        return f"{self.server_url}/api/{endpoint}"

    def save_settings(self):
        """Save current settings to appropriate location"""
        if self.is_local:
            # We're in TGWUI mode, we don't need to save to AllTalk's config
            # The settings are maintained by TGWUI
            pass
        else:
            # In remote mode, save to local config file
            settings = {
                "branding": self.branding,
                "tgwui": self.config["tgwui"],
                "remote_connection": {
                    "server_protocol": self.server["protocol"],
                    "server_address": self.server["address"],
                    "connection_timeout": self.server["timeout"],
                    "use_legacy_api": self.config['remote_connection']['use_legacy_api']
                }
            }
            config_path = Path(__file__).parent / "tgwui_remote_config.json"
            with open(config_path, "w") as f:
                json.dump(settings, f, indent=4)           

# Initialize mode manager at startup
mode_manager = TGWUIModeManager()

##########################
# Central print function #
##########################
def print_func(message, message_type="standard", component="TTS", debug_type=None):
    """Centralized print function for AllTalk messages
    
    Args:
        message (str): The message to print
        message_type (str): Type of message ("standard", "warning", "error", "debug")
        component (str): Component identifier ("TTS", "ENG", "Server", etc.)
        debug_type (str, optional): Can be supplied on the call to specify things like:
            - The function that is sending the debug request.
            - The operation taking place
    """
    prefix = f"[{mode_manager.branding}{component}] "

    if message_type == "warning":
        print(f"{prefix}\033[91mWarning\033[0m {message}")
    elif message_type == "error":
        print(f"{prefix}\033[91mError\033[0m {message}")
    elif message_type == "debug":
        if debug_type is None:
            debug_type = ""
        color = "\033[92m" if "Function entry" in message else "\033[93m"            
        print(f"{prefix}\033[94mDebug\033[0m {color}{debug_type}\033[0m {message}")
    else:  # standard message
        print(f"{prefix}{message}")

def debug_func_entry():
    """Print debug message for function entry if debug_func is enabled/true"""
    if mode_manager.debug_func:
        print_func("Function entry", "debug", debug_type=inspect.currentframe().f_back.f_code.co_name)

# Load languages
with open(Path(__file__).parent / "languages.json", encoding="utf8") as f:
    languages = json.load(f)

# Setup other globals
process_lock = threading.Lock()
models_available = None
alltalk_settings = None
current_model_loaded = None
tts_model_loaded = None

def get_alltalk_settings():
    if mode_manager.debug_func: 
        print_func("Function entry", "debug", debug_type=inspect.currentframe().f_code.co_name)
    global current_model_loaded, models_available
    
    def log_error(message, status_code=None):
        """Helper function for consistent error logging"""
        if status_code:
            print_func(f"{message} Status code:\n{status_code}", message_type="warning")
        else:
            print_func(f"{message}", message_type="warning")

    try:
        # Make all API requests
        api_calls = {
            "voices": requests.get(
                mode_manager.get_api_url("voices"),
                timeout=mode_manager.server["timeout"]
            ),
            "rvc": requests.get(
                mode_manager.get_api_url("rvcvoices"),
                timeout=mode_manager.server["timeout"]
            ),
            "settings": requests.get(
                mode_manager.get_api_url("currentsettings"),
                timeout=mode_manager.server["timeout"]
            )
        }

        # Check if all responses are successful
        if all(response.status_code == 200 for response in api_calls.values()):
            try:
                voices_data = api_calls["voices"].json()
                rvcvoices_data = api_calls["rvc"].json()
                settings_data = api_calls["settings"].json()

                # Update global variables
                models_available = [model["name"] for model in settings_data["models_available"]]
                current_model_loaded = settings_data["current_model_loaded"]
                
                # Update our local config with received values
                if mode_manager.is_local:
                    mode_manager.config["tgwui"].update({
                        "tgwui_character_voice": voices_data["voices"][0] if voices_data["voices"] else "Please Refresh Settings",
                        "tgwui_narrator_voice": voices_data["voices"][0] if voices_data["voices"] else "Please Refresh Settings",
                        "tgwui_rvc_char_voice": rvcvoices_data["rvcvoices"][0] if rvcvoices_data["rvcvoices"] else "Disabled",
                        "tgwui_rvc_narr_voice": rvcvoices_data["rvcvoices"][0] if rvcvoices_data["rvcvoices"] else "Disabled",
                    })                

                return AllTalkServerSettings.from_api_response(
                    voices_data,
                    rvcvoices_data,
                    settings_data
                )
            except json.JSONDecodeError as e:
                log_error(f"Failed to decode JSON response: {e}")
                return AllTalkServerSettings()
        else:
            # Log specific failures
            log_error(f"Failed to retrieve {mode_manager.branding}settings from API.")
            for name, response in api_calls.items():
                if response.status_code != 200:
                    log_error(f"Failed to retrieve {name} from API", response.status_code)
            return AllTalkServerSettings()

    except (RequestException, ConnectionError) as e:
        log_error(f"Unable to connect to the {mode_manager.branding}server", str(e))
        return AllTalkServerSettings()

class AllTalkServerSettings:
    def __init__(self):
        # Voice and model settings
        self.voices = ["Please Refresh Settings"]
        self.rvcvoices = ["Please Refresh Settings"]
        self.models_available = ["Please Refresh Settings"]
        self.current_model_loaded = "Please Refresh Settings"
        self.manufacturer_name = ""
        
        # Engine capabilities
        self.deepspeed_capable = False
        self.deepspeed_available = False
        self.deepspeed_enabled = False
        self.languages_capable = False
        self.multivoice_capable = False
        self.multimodel_capable = False
        self.streaming_capable = False
        self.ttsengines_installed = False
        
        # Generation settings
        self.generationspeed_capable = False
        self.generationspeed_set = 1.0
        self.lowvram_capable = False
        self.lowvram_enabled = False
        self.pitch_capable = False
        self.pitch_set = 0
        self.repetitionpenalty_capable = False
        self.repetitionpenalty_set = 10.0
        self.temperature_capable = False
        self.temperature_set = 0.75

    @classmethod
    def from_api_response(cls, voices_data, rvcvoices_data, settings_data):
        """Create settings instance from API response data"""
        debug_func_entry()        
        if mode_manager.debug_func:
            print_func("Function entry", "debug", debug_type=inspect.currentframe().f_code.co_name)
        settings = cls()
        
        # Update voice and model information
        settings.voices = sorted(voices_data["voices"])
        settings.rvcvoices = rvcvoices_data["rvcvoices"]
        settings.models_available = [model["name"] for model in settings_data["models_available"]]
        settings.current_model_loaded = settings_data["current_model_loaded"]
        settings.manufacturer_name = settings_data.get("manufacturer_name", "")
        
        # Update capabilities and settings
        settings_mapping = {
            # Capabilities
            "deepspeed_capable": ("deepspeed_capable", False),
            "deepspeed_available": ("deepspeed_available", False),
            "deepspeed_enabled": ("deepspeed_enabled", False),
            "languages_capable": ("languages_capable", False),
            "multivoice_capable": ("multivoice_capable", False),
            "multimodel_capable": ("multimodel_capable", False),
            "streaming_capable": ("streaming_capable", False),
            "ttsengines_installed": ("ttsengines_installed", False),
            
            # Generation settings
            "generationspeed_capable": ("generationspeed_capable", False),
            "generationspeed_set": ("generationspeed_set", 1.0),
            "lowvram_capable": ("lowvram_capable", False),
            "lowvram_enabled": ("lowvram_enabled", False),
            "pitch_capable": ("pitch_capable", False),
            "pitch_set": ("pitch_enabled", 0),
            "repetitionpenalty_capable": ("repetitionpenalty_capable", False),
            "repetitionpenalty_set": ("repetitionpenalty_set", 10.0),
            "temperature_capable": ("temperature_capable", False),
            "temperature_set": ("temperature_set", 0.75),
        }

        # Update all settings from the mapping
        for attr, (key, default) in settings_mapping.items():
            setattr(settings, attr, settings_data.get(key, default))
        
        return settings


# Print Spashscreen to console if running as Remote Extension
if not mode_manager.is_local: 
    print_func("\033[94m     _    _ _ \033[1;35m_____     _ _     \033[0m  _____ _____ ____  ")# pylint: disable=line-too-long anomalous-backslash-in-string
    print_func("\033[94m    / \  | | |\033[1;35m_   _|_ _| | | __ \033[0m |_   _|_   _/ ___| ")# pylint: disable=line-too-long anomalous-backslash-in-string
    print_func("\033[94m   / _ \ | | |\033[1;35m | |/ _` | | |/ / \033[0m   | |   | | \___ \ ")# pylint: disable=line-too-long anomalous-backslash-in-string
    print_func("\033[94m  / ___ \| | |\033[1;35m | | (_| | |   <  \033[0m   | |   | |  ___) |")# pylint: disable=line-too-long anomalous-backslash-in-string
    print_func("\033[94m /_/   \_\_|_|\033[1;35m |_|\__,_|_|_|\_\ \033[0m   |_|   |_| |____/ ")# pylint: disable=line-too-long anomalous-backslash-in-string
    print_func("")
    print_func(f"\033[92m{mode_manager.branding}startup Mode   : \033[93mText-Gen-webui Remote\033[0m")
    print_func("")

def stop_generate_tts():
    """Sends request to stop current TTS generation"""
    debug_func_entry()
    api_url = mode_manager.get_api_url("stop-generation")
    
    if mode_manager.debug_tgwui:
        print_func("Attempting to stop TTS generation", message_type="debug", debug_type="stop_generate_tts")
    
    try:
        response = requests.put(api_url, timeout=mode_manager.server["timeout"])
        if response.status_code == 200:
            return response.json()["message"]
        else:
            print_func(f"Failed to stop generation. Status code:\n{response.status_code}", message_type="warning")
            return {"message": "Failed to stop generation"}
    except (RequestException, ConnectionError) as e:
        print_func(f"Unable to connect to the {mode_manager.branding}server. Status code:\n{str(e)}", message_type="warning")
        return {"message": "Failed to stop generation"}

def send_reload_request(value_sent):
    """Requests AllTalk server to load a different TTS model"""
    debug_func_entry()
    global tts_model_loaded
    if not process_lock.acquire(blocking=False):
        return {"status": "error", "message": "Server is currently busy. Try again in a few seconds."}    
    try:
        tts_model_loaded = False        
        url = mode_manager.get_api_url("reload")
        payload = {"tts_method": value_sent}
        
        if mode_manager.debug_tgwui:
            print_func(f"Requesting model reload to: {value_sent}", message_type="debug", debug_type="send_reload_request")
        
        response = requests.post(url, params=payload)
        response.raise_for_status()
        tts_model_loaded = True        
    except requests.exceptions.RequestException as e:
        print_func(f"Error during request to webserver process: Status code:\n{e}", message_type="warning")
        tts_model_loaded = True
        return {"status": "error", "message": str(e)}
    finally:
        process_lock.release()

def send_lowvram_request(value_sent):
    """Toggles low VRAM mode in AllTalk server"""
    debug_func_entry()
    global tts_model_loaded
    if not process_lock.acquire(blocking=False):
        return {"status": "error", "message": "Server is currently busy. Try again in a few seconds."}
    try:
        tts_model_loaded = False
        audio_path = this_dir / ("lowvramenabled.wav" if value_sent else "lowvramdisabled.wav")
        
        if mode_manager.debug_tgwui:
            print_func(f"Setting low VRAM mode to: {value_sent}", message_type="debug", debug_type="send_lowvram_request")
        
        url = mode_manager.get_api_url(f"lowvramsetting?new_low_vram_value={value_sent}")
        response = requests.post(url, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        json_response = response.json()
        if json_response.get("status") == "lowvram-success":
            tts_model_loaded = True
            
        return f'<audio src="file/{audio_path}" controls autoplay></audio>'
    except requests.exceptions.RequestException as e:
        print_func(f"Error during request to webserver process: Status code:\n{e}", message_type="warning")
        return {"status": "error", "message": str(e)}
    finally:
        process_lock.release()    

def send_deepspeed_request(value_sent):
    """Toggles DeepSpeed acceleration in AllTalk server"""
    debug_func_entry()
    global tts_model_loaded
    if not process_lock.acquire(blocking=False):
        return {"status": "error", "message": "Server is currently busy. Try again in a few seconds."}
    try:
        tts_model_loaded = False
        audio_path = this_dir / ("deepspeedenabled.wav" if value_sent else "deepspeeddisabled.wav")
        
        if mode_manager.debug_tgwui:
            print_func(f"Setting DeepSpeed to: {value_sent}", message_type="debug", debug_type="send_deepspeed_request")
        
        url = mode_manager.get_api_url(f"deepspeed?new_deepspeed_value={value_sent}")
        response = requests.post(url, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        json_response = response.json()
        if json_response.get("status") == "deepspeed-success":
            tts_model_loaded = True
        
        process_lock.release()
        return f'<audio src="file/{audio_path}" controls autoplay></audio>'
    except requests.exceptions.RequestException as e:
        print_func(f"Error during request to webserver process: Status code:\n{e}", message_type="warning")
        return {"status": "error", "message": str(e)}
    finally:
        process_lock.release()

def send_and_generate(gen_text, gen_character_voice, gen_rvccharacter_voice, gen_rvccharacter_pitch, gen_narrator_voice, gen_rvcnarrator_voice, 
                     gen_rvcnarrator_pitch, gen_narrator_activated, gen_textnotinisde, gen_repetition, gen_language, gen_filter, gen_speed, 
                     gen_pitch, gen_autoplay, gen_autoplay_vol, gen_file_name, gen_temperature, gen_filetimestamp, 
                     gen_stream, gen_stopcurrentgen):
    """Sends text to AllTalk server for TTS generation"""
    debug_func_entry()
    api_url = mode_manager.get_api_url("tts-generate")
    
    if mode_manager.debug_tgwui:
        print_func("Starting TTS generation request", message_type="debug", debug_type="send_and_generate")
    
    if gen_stopcurrentgen:
        stop_generate_tts()
        
    mode_manager.save_settings()

    if gen_stream == "true":
        api_url = mode_manager.get_api_url("tts-generate-streaming")
        encoded_text = requests.utils.quote(gen_text)
        streaming_url = f"{api_url}?text={encoded_text}&voice={gen_character_voice}&language={gen_language}&output_file={gen_file_name}"
        return streaming_url, str("TTS Streaming Audio Generated")
    
    data = {
        "text_input": gen_text,
        "text_filtering": gen_filter,
        "character_voice_gen": gen_character_voice,
        "rvccharacter_voice_gen": gen_rvccharacter_voice,
        "rvccharacter_pitch": gen_rvccharacter_pitch,
        "narrator_enabled": str(gen_narrator_activated).lower(),
        "narrator_voice_gen": gen_narrator_voice,
        "rvcnarrator_voice_gen": gen_rvcnarrator_voice,
        "rvcnarrator_pitch": gen_rvcnarrator_pitch,
        "text_not_inside": gen_textnotinisde,
        "language": gen_language,
        "output_file_name": str(gen_file_name),
        "output_file_timestamp": str(gen_filetimestamp).lower(),
        "autoplay": str(gen_autoplay).lower(),
        "autoplay_volume": str(gen_autoplay_vol),
        "speed": str(gen_speed),
        "pitch": str(gen_pitch),
        "temperature": str(gen_temperature),
        "repetition_penalty": str(gen_repetition),
    }
    
    if mode_manager.debug_tgwui:
        print_func("Generation request parameters:", message_type="debug", debug_type="send_and_generate")
        for key, value in data.items():
            print_func(f"{key}: {value}", message_type="debug", debug_type="send_and_generate")
    
    try:
        response = requests.post(api_url, data=data)
        response.raise_for_status()
        result = response.json()
        
        if gen_autoplay == "true":
            return None, str("TTS Audio Generated (Played remotely)")
            
        if mode_manager.config["remote_connection"]["use_legacy_api"]:
            return result['output_file_url'], str("TTS Audio Generated")
        else:
            output_file_url = f"{mode_manager.server_url}{result['output_file_url']}"
            return output_file_url, str("TTS Audio Generated")
    except (RequestException, ConnectionError) as e:
        print_func(f"Error occurred during the API request: Status code:\n{str(e)}", message_type="warning")
        return None, str("Error occurred during the API request") 

def output_modifier(string, state):
    """Modifies TGWUI output to include TTS audio"""
    debug_func_entry()
    if not mode_manager.config["tgwui"]["tgwui_activate_tts"]:
        return string

    if mode_manager.debug_tgwui:
        print_func("Processing text for TTS generation", message_type="debug", debug_type="output_modifier")
        
    # Strip out Images
    img_info = ""
    cleaned_text, img_info = tgwui_extract_and_remove_images(string)
    if cleaned_text is None:
        return

    # Get current settings
    language_code = languages.get(mode_manager.config["tgwui"]["tgwui_language"])
    character_voice = mode_manager.config["tgwui"]["tgwui_character_voice"]
    rvc_character_voice = mode_manager.config["tgwui"]["tgwui_rvc_char_voice"]
    rvc_character_pitch = mode_manager.config["tgwui"]["tgwui_rvc_char_pitch"]
    narrator_voice = mode_manager.config["tgwui"]["tgwui_narrator_voice"]
    rvc_narrator_voice = mode_manager.config["tgwui"]["tgwui_rvc_narr_voice"]
    rvc_narrator_pitch = mode_manager.config["tgwui"]["tgwui_rvc_narr_pitch"]
    narrator_enabled = mode_manager.config["tgwui"]["tgwui_narrator_enabled"]
    text_not_inside = mode_manager.config["tgwui"]["tgwui_non_quoted_text_is"]
    repetition_policy = mode_manager.config["tgwui"]["tgwui_repetitionpenalty_set"]
    speed = mode_manager.config["tgwui"]["tgwui_generationspeed_set"]
    pitch = mode_manager.config["tgwui"]["tgwui_pitch_set"]
    temperature = mode_manager.config["tgwui"]["tgwui_temperature_set"]

    if mode_manager.debug_tgwui:
        print_func(f"Using character voice: {character_voice}, narrator voice: {narrator_voice}", 
                  message_type="debug", debug_type="output_modifier")
        print_func(f"Text length: {len(cleaned_text)} characters", 
                  message_type="debug", debug_type="output_modifier")

    # Lock and process TTS request
    if process_lock.acquire(blocking=False):
        try:
            output_file = state["character_menu"] if "character_menu" in state else str("TTSOUT_")
            # Ensure the file name matches the allowed pattern of AllTalk
            output_file = re.sub(r"[^a-zA-Z0-9_]", "", output_file)
            
            generate_response, status_message = send_and_generate(
                cleaned_text, character_voice, rvc_character_voice, rvc_character_pitch, narrator_voice, 
                rvc_narrator_voice, rvc_narrator_pitch, narrator_enabled, text_not_inside, repetition_policy,
                language_code, "html", speed, pitch, False, 0.8, output_file, 
                temperature, True, False, False
            )

            if status_message == "TTS Audio Generated":
                autoplay = "autoplay" if mode_manager.config["tgwui"]["tgwui_autoplay_tts"] else ""
                string = f'<audio src="{generate_response}" controls {autoplay}></audio>'
                
                if mode_manager.config["tgwui"]["tgwui_show_text"]:
                    string += tgwui_reinsert_images(cleaned_text, img_info)
                    
                shared.processing_message = "*Is typing...*"
                return string
            else:
                print_func(f"Audio generation failed. Status code:\n{status_message}", message_type="warning")
        finally:
            process_lock.release()
    else:
        print_func("Audio generation is already in progress. Please wait.", message_type="warning")
        return

@debounce
def voice_preview(string):
    """Generates a preview of the selected voice settings"""
    # debug_func_entry()
    print("in voice preview function")
    if not mode_manager.config["tgwui"]["tgwui_activate_tts"]:
        return string

    if mode_manager.debug_tgwui:
        print_func("Generating voice preview", message_type="debug", debug_type="voice_preview")

    language_code = languages.get(mode_manager.config["tgwui"]["tgwui_language"])
    if not string:
        string = random_sentence()
        
    if mode_manager.debug_tgwui:
        print_func(f"Preview text: {string}", message_type="debug", debug_type="voice_preview")

    generate_response, status_message = send_and_generate(
        string, 
        mode_manager.config["tgwui"]["tgwui_character_voice"],
        mode_manager.config["tgwui"]["tgwui_rvc_char_voice"],
        mode_manager.config["tgwui"]["tgwui_rvc_char_pitch"],
        mode_manager.config["tgwui"]["tgwui_narrator_voice"],
        mode_manager.config["tgwui"]["tgwui_rvc_narr_voice"],
        mode_manager.config["tgwui"]["tgwui_rvc_narr_pitch"],
        False, "character",
        mode_manager.config["tgwui"]["tgwui_repetitionpenalty_set"],
        language_code, "standard",
        mode_manager.config["tgwui"]["tgwui_generationspeed_set"],
        mode_manager.config["tgwui"]["tgwui_pitch_set"],
        False, 0.8, "previewvoice",
        mode_manager.config["tgwui"]["tgwui_temperature_set"],
        False, False, False
    )

    if status_message == "TTS Audio Generated":
        autoplay = "autoplay" if mode_manager.config["tgwui"]["tgwui_autoplay_tts"] else ""
        return f'<audio src="{generate_response}?{int(time.time())}" controls {autoplay}></audio>'
    else:
        return f"[{mode_manager.branding}Server] Audio generation failed. Status code:\n{status_message}"

img_pattern = r'<img[^>]*src\s*=\s*["\'][^"\'>]+["\'][^>]*>'

def tgwui_extract_and_remove_images(text):
    """Extracts and removes image tags from text for clean TTS processing"""
    debug_func_entry()
    img_matches = re.findall(img_pattern, text)
    img_info = "\n".join(img_matches)
    cleaned_text = re.sub(img_pattern, '', text)
    
    if mode_manager.debug_tgwui:
        print_func(f"Found {len(img_matches)} images in text", message_type="debug", debug_type="extract_images")
        
    return cleaned_text, img_info

def tgwui_reinsert_images(text, img_info):
    """Reinserts the previously extracted image data back into the text"""
    debug_func_entry()
    if img_info:  # Check if there are images to reinsert
        text += f"\n\n{img_info}"
    return text

################################################################
# TGWUI # Used to generate a preview voice sample within TGWUI #
################################################################
def random_sentence():
    """Returns a random sentence from Harvard sentences file for voice preview"""
    debug_func_entry()
    with open(this_dir / "harvard_sentences.txt") as f:
        return random.choice(list(f)).rstrip()

###################################################################
# TGWUI # Used to inform TGWUI that TTS is disabled/not activated #
###################################################################
def state_modifier(state):
    """Modifies TGWUI state to disable streaming when TTS is active"""
    debug_func_entry()
    if not mode_manager.config["tgwui"]["tgwui_activate_tts"]:
        return state
    
    if mode_manager.debug_tgwui:
        print_func("Disabling streaming for TTS", message_type="debug", debug_type="state_modifier")
        
    state["stream"] = False
    return state

###################################################################
# TGWUI #  Sends message to TGWUI interface during TTS generation #
###################################################################
def input_modifier(string, state):
    """Updates TGWUI interface message during TTS processing"""
    debug_func_entry()
    if not mode_manager.config["tgwui"]["tgwui_activate_tts"]:
        return string
        
    if mode_manager.debug_tgwui:
        print_func("Setting processing message for TTS", message_type="debug", debug_type="input_modifier")
        
    shared.processing_message = "*Is recording a voice message...*"
    return string

########################################################################
# TGWUI # Used to delete historic TTS audios from TGWUI chat interface #
########################################################################
def remove_tts_from_history(history):
    '''Removes audio from the chat with TGWUI's chat interface'''
    debug_func_entry()
    for i, entry in enumerate(history["internal"]):
        history["visible"][i] = [history["visible"][i][0], entry[1]]
    return history

def toggle_text_in_history(history):
    '''Hides text from the chat with TGWUI's chat interface'''
    debug_func_entry()
    for i, entry in enumerate(history["visible"]):
        visible_reply = entry[1]
        if visible_reply.startswith("<audio"):
            if mode_manager.config["tgwui"]["tgwui_show_text"]:
                reply = history["internal"][i][1]
                history["visible"][i] = [
                    history["visible"][i][0],
                    f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}",
                ]
            else:
                history["visible"][i] = [
                    history["visible"][i][0],
                    f"{visible_reply.split('</audio>')[0]}</audio>",
                ]
    return history

def history_modifier(history):
    '''Controls/Enables autoplay of TTS in TGWUI interface'''
    debug_func_entry()
    # Remove autoplay from the last reply
    if len(history["internal"]) > 0:
        history["visible"][-1] = [
            history["visible"][-1][0],
            history["visible"][-1][1].replace("controls autoplay>", "controls>"),
        ]
    return history

def tgwui_update_alltalk_protocol(value_sent):
    """Updates the server protocol setting"""
    debug_func_entry()
    if mode_manager.debug_tgwui:
        print_func(f"Updating server protocol to: {value_sent}", 
                  message_type="debug", debug_type="update_protocol")
    mode_manager.server["protocol"] = value_sent
    
def tgwui_update_alltalk_ip_port(value_sent):
    """Updates the server address settings"""
    debug_func_entry()
    if mode_manager.debug_tgwui:
        print_func(f"Updating server address to: {value_sent}", 
                  message_type="debug", debug_type="update_ip_port")
    mode_manager.server["address"] = value_sent

def tgwui_handle_ttsmodel_dropdown_change(model_name):
    """Handles model changes from Gradio dropdown, with debounce protection"""
    debug_func_entry()
    if not getattr(tgwui_handle_ttsmodel_dropdown_change, "skip_reload", False):
        if mode_manager.debug_tgwui:
            print_func(f"Switching model to: {model_name}", message_type="debug", debug_type="model_change")
        send_reload_request(model_name)
    else:
        if mode_manager.debug_tgwui:
            print_func("Skipping model reload (debounced)", message_type="debug", debug_type="model_change")
        tgwui_handle_ttsmodel_dropdown_change.skip_reload = False

def tgwui_update_dropdowns():
    """Updates all Gradio dropdowns with current server settings"""
    debug_func_entry()
    global at_settings
    
    if mode_manager.debug_tgwui:
        print_func("Refreshing server settings", message_type="debug", debug_type="update_dropdowns")
        
    at_settings = get_alltalk_settings()

    if mode_manager.debug_tgwui:
        print_func(f"Retrieved {len(at_settings.voices)} voices", message_type="debug", debug_type="update_dropdowns")
        print_func(f"Retrieved {len(at_settings.rvcvoices)} RVC voices", message_type="debug", debug_type="update_dropdowns")
        print_func(f"Current model: {at_settings.current_model_loaded}", message_type="debug", debug_type="update_dropdowns")

    current_voices = at_settings.voices
    rvccurrent_voices = at_settings.rvcvoices
    rvccurrent_character_voice = mode_manager.config["tgwui"]["tgwui_rvc_char_voice"]
    rvccurrent_character_pitch = mode_manager.config["tgwui"]["tgwui_rvc_char_pitch"]
    rvccurrent_narrator_voice = mode_manager.config["tgwui"]["tgwui_rvc_char_voice"]
    rvccurrent_narrator_pitch = mode_manager.config["tgwui"]["tgwui_rvc_char_pitch"]
    current_models_available = sorted(at_settings.models_available)
    current_model_loaded = at_settings.current_model_loaded
    current_character_voice = mode_manager.config["tgwui"]["tgwui_character_voice"]
    current_narrator_voice = mode_manager.config["tgwui"]["tgwui_narrator_voice"]

    # Capability checks
    current_lowvram_capable = at_settings.lowvram_capable
    current_lowvram_enabled = at_settings.lowvram_enabled
    current_temperature_capable = at_settings.temperature_capable
    current_repetitionpenalty_capable = at_settings.repetitionpenalty_capable
    current_generationspeed_capable = at_settings.generationspeed_capable
    current_pitch_capable = at_settings.pitch_capable
    current_deepspeed_capable = at_settings.deepspeed_capable
    current_deepspeed_enabled = at_settings.deepspeed_enabled
    current_non_quoted_text_is = mode_manager.config["tgwui"]["tgwui_non_quoted_text_is"]
    current_languages_capable = at_settings.languages_capable

    # Set appropriate labels based on capabilities
    language_label = "Languages" if at_settings.languages_capable else "Model not multi language"

    # Update voice selections if needed
    if current_character_voice not in current_voices:
        if mode_manager.debug_tgwui:
            print_func(f"Character voice {current_character_voice} not found, resetting", 
                      message_type="debug", debug_type="update_dropdowns")
        current_character_voice = current_voices[0] if current_voices else ""
        mode_manager.config["tgwui"]["tgwui_character_voice"] = current_character_voice

    if current_narrator_voice not in current_voices:
        if mode_manager.debug_tgwui:
            print_func(f"Narrator voice {current_narrator_voice} not found, resetting", 
                      message_type="debug", debug_type="update_dropdowns")
        current_narrator_voice = current_voices[0] if current_voices else ""
        mode_manager.config["tgwui"]["tgwui_narrator_voice"] = current_narrator_voice

    # Update RVC voice selections if needed
    if rvccurrent_character_voice not in rvccurrent_voices:
        if mode_manager.debug_tgwui:
            print_func(f"RVC character voice {rvccurrent_character_voice} not found, resetting", 
                      message_type="debug", debug_type="update_dropdowns")
        rvccurrent_character_voice = rvccurrent_voices[0] if rvccurrent_voices else ""
        mode_manager.config["tgwui"]["tgwui_rvc_char_voice"] = rvccurrent_character_voice

    if rvccurrent_narrator_voice not in rvccurrent_voices:
        if mode_manager.debug_tgwui:
            print_func(f"RVC narrator voice {rvccurrent_narrator_voice} not found, resetting", 
                      message_type="debug", debug_type="update_dropdowns")
        rvccurrent_narrator_voice = rvccurrent_voices[0] if rvccurrent_voices else ""
        mode_manager.config["tgwui"]["tgwui_rvc_char_voice"] = rvccurrent_narrator_voice

    rvccurrent_character_pitch = rvccurrent_character_pitch if rvccurrent_character_pitch else 0
    rvccurrent_narrator_pitch = rvccurrent_narrator_pitch if rvccurrent_narrator_pitch else 0

    # Prevent model reload during update
    tgwui_handle_ttsmodel_dropdown_change.skip_reload = True

    # Return updated Gradio components
    return_values = [
        gr.Checkbox(interactive=current_lowvram_capable, value=current_lowvram_enabled),
        gr.Checkbox(interactive=current_deepspeed_capable, value=current_deepspeed_enabled),
        gr.Dropdown(choices=current_voices, value=current_character_voice),
        gr.Dropdown(choices=rvccurrent_voices, value=rvccurrent_character_voice, interactive=True),
        gr.Slider(value=rvccurrent_character_pitch),       
        gr.Dropdown(choices=current_voices, value=current_narrator_voice),
        gr.Dropdown(choices=rvccurrent_voices, value=rvccurrent_narrator_voice, interactive=True),
        gr.Slider(value=rvccurrent_narrator_pitch),       
        gr.Dropdown(choices=current_models_available, value=current_model_loaded),
        gr.Dropdown(interactive=current_temperature_capable),
        gr.Dropdown(interactive=current_repetitionpenalty_capable),
        gr.Dropdown(interactive=current_languages_capable, label=language_label),
        gr.Dropdown(interactive=current_generationspeed_capable),
        gr.Dropdown(interactive=current_pitch_capable),
        gr.Dropdown(value=current_non_quoted_text_is),
    ]
    
    def reset_skip_reload():
        tgwui_handle_ttsmodel_dropdown_change.skip_reload = False
    
    threading.Timer(0.5, reset_skip_reload).start()
    
    return return_values

def ui():
    """Creates the TGWUI Gradio interface"""
    global alltalk_settings
    
    if mode_manager.debug_tgwui:
        print_func("Initializing TGWUI interface", message_type="debug", debug_type="ui")
    
    alltalk_settings = get_alltalk_settings()

    with gr.Accordion(mode_manager.branding + " TTS (Text-gen-webui Remote)"):
        tgwui_available_voices_gr = alltalk_settings.voices
        tgwui_rvc_available_voices_gr = alltalk_settings.rvcvoices
        
        # Activate alltalk_tts, Enable autoplay, Show text
        with gr.Row():
            tgwui_activate_tts_gr = gr.Checkbox(
                value=mode_manager.config["tgwui"]["tgwui_activate_tts"],
                label="Enable TGWUI TTS"
            )
            tgwui_autoplay_gr = gr.Checkbox(
                value=mode_manager.config["tgwui"]["tgwui_autoplay_tts"],
                label="Autoplay TTS Generated"
            )
            tgwui_show_text_gr = gr.Checkbox(
                value=mode_manager.config["tgwui"]["tgwui_show_text"],
                label="Show Text in chat"
            )

        # Low vram enable, Deepspeed enable, Link
        with gr.Row():
            tgwui_lowvram_enabled_gr = gr.Checkbox(
                value=alltalk_settings.lowvram_enabled if alltalk_settings.lowvram_capable else False,
                label="Enable Low VRAM Mode",
                interactive=alltalk_settings.lowvram_capable
            )
            tgwui_lowvram_enabled_play_gr = gr.HTML(visible=False)
            
            tgwui_deepspeed_enabled_gr = gr.Checkbox(
                value=mode_manager.config["tgwui"]["tgwui_deepspeed_enabled"],
                label="Enable DeepSpeed",
                interactive=alltalk_settings.deepspeed_capable
            )
            tgwui_deepspeed_enabled_play_gr = gr.HTML(visible=False)
            
            tgwui_empty_space_gr = gr.HTML(
                f"<p><a href='{mode_manager.server_url}'>AllTalk Server & Documentation Link</a></p>"
            )
            
            # Model, Language, Character voice
        with gr.Row():
            tgwui_tts_dropdown_gr = gr.Dropdown(
                choices=models_available,
                label="TTS Engine/Model",
                value=current_model_loaded
            )
            tgwui_language_gr = gr.Dropdown(
                choices=languages.keys(),
                label="Languages" if alltalk_settings.languages_capable else "Model not multi language",
                interactive=alltalk_settings.languages_capable,
                value=mode_manager.config["tgwui"]["tgwui_language"]
            )
            tgwui_narrator_enabled_gr = gr.Dropdown(
                choices=[("Enabled", "true"), ("Disabled", "false"), ("Silent", "silent")],
                label="Narrator Enable",
                value="true" if mode_manager.config.get("tgwui_narrator_enabled") == "true" 
                      else ("silent" if mode_manager.config.get("tgwui_narrator_enabled") == "silent" 
                      else "false")
            )

        # Narrator voice settings
        with gr.Row():
            tgwui_default_voice_gr = mode_manager.config["tgwui"]["tgwui_character_voice"]
            if tgwui_default_voice_gr not in tgwui_available_voices_gr:
                tgwui_default_voice_gr = tgwui_available_voices_gr[0] if tgwui_available_voices_gr else ""
            
            tgwui_character_voice_gr = gr.Dropdown(
                choices=tgwui_available_voices_gr,
                label="Character Voice",
                value=tgwui_default_voice_gr,
                allow_custom_value=True
            )
            
            tgwui_narr_voice_gr = mode_manager.config["tgwui"]["tgwui_narrator_voice"]
            if tgwui_narr_voice_gr not in tgwui_available_voices_gr:
                tgwui_narr_voice_gr = tgwui_available_voices_gr[0] if tgwui_available_voices_gr else ""
            
            tgwui_narrator_voice_gr = gr.Dropdown(
                choices=tgwui_available_voices_gr,
                label="Narrator Voice",
                value=tgwui_narr_voice_gr,
                allow_custom_value=True
            )
            
            tgwui_non_quoted_text_is_gr = gr.Dropdown(
                choices=[("Character", "character"), ("Narrator", "narrator"), ("Silent", "silent")],
                label='Narrator unmarked text is',
                value=mode_manager.config.get("tgwui_non_quoted_text_is", "character")
            )

        # RVC voices
        with gr.Row():
            tgwui_rvc_default_voice_gr = mode_manager.config["tgwui"]["tgwui_rvc_char_voice"]
            tgwui_rvc_narrator_voice_gr = mode_manager.config["tgwui"]["tgwui_rvc_narr_voice"]
            
            if tgwui_rvc_default_voice_gr not in tgwui_rvc_available_voices_gr:
                tgwui_rvc_default_voice_gr = tgwui_rvc_available_voices_gr[0] if tgwui_rvc_available_voices_gr else ""
            
            tgwui_rvc_char_voice_gr = gr.Dropdown(
                choices=tgwui_rvc_available_voices_gr,
                label="RVC Character Voice",
                value=tgwui_rvc_default_voice_gr,
                allow_custom_value=True
            )
            
            if tgwui_rvc_narrator_voice_gr not in tgwui_rvc_available_voices_gr:
                tgwui_rvc_narrator_voice_gr = tgwui_rvc_available_voices_gr[0] if tgwui_rvc_available_voices_gr else ""
            
            tgwui_rvc_narr_voice_gr = gr.Dropdown(
                choices=tgwui_rvc_available_voices_gr,
                label="RVC Narrator Voice",
                value=tgwui_rvc_narrator_voice_gr,
                allow_custom_value=True
            )
            
        # RVC pitch control
        with gr.Row():
            tgwui_rvc_char_pitch_gr = gr.Slider(
                minimum=-24,
                maximum=+24,
                step=1,
                label="RVC Character Pitch",
                value=mode_manager.config["tgwui"]["tgwui_rvc_char_pitch"],
            )
            tgwui_rvc_narr_pitch_gr = gr.Slider(
                minimum=-24,
                maximum=+24,
                step=1,
                label="RVC Narrator Pitch",
                value=mode_manager.config["tgwui"]["tgwui_rvc_narr_pitch"],
            )            

        # Generation settings
        with gr.Row():
            tgwui_temperature_set_gr = gr.Slider(
                minimum=0.05,
                maximum=1,
                step=0.05,
                label="Temperature",
                value=mode_manager.config["tgwui"]["tgwui_temperature_set"],
                interactive=alltalk_settings.temperature_capable
            )
            tgwui_repetitionpenalty_set_gr = gr.Slider(
                minimum=0.5,
                maximum=20,
                step=0.5,
                label="Repetition Penalty",
                value=mode_manager.config["tgwui"]["tgwui_repetitionpenalty_set"],
                interactive=alltalk_settings.repetitionpenalty_capable
            )
        with gr.Row():
            tgwui_generationspeed_set_gr = gr.Slider(
                minimum=0.30,
                maximum=2.00,
                step=0.10,
                label="TTS Speed",
                value=mode_manager.config["tgwui"]["tgwui_generationspeed_set"],
                interactive=alltalk_settings.generationspeed_capable
            )
            tgwui_pitch_set_gr = gr.Slider(
                minimum=-10,
                maximum=10,
                step=1,
                label="Pitch",
                value=mode_manager.config["tgwui"]["tgwui_pitch_set"],
                interactive=alltalk_settings.pitch_capable
            )

        # Preview section
        with gr.Row():
            tgwui_preview_text_gr = gr.Text(
                show_label=False,
                placeholder="Preview text",
                elem_id="silero_preview_text",
                scale=2
            )
            tgwui_preview_play_gr = gr.Button(
                "Generate Preview",
                scale=1
            )
            tgwui_preview_audio_gr = gr.HTML(visible=False)

        # Server connection settings (only shown in remote mode)
        if not mode_manager.is_local:
            with gr.Row():
                tgwui_protocol_gr = gr.Dropdown(
                    choices=["http://", "https://"],
                    label="AllTalk Server Protocol",
                    value=mode_manager.server["protocol"]
                )
                tgwui_ip_address_port_gr = gr.Textbox(
                    label="AllTalk Server IP:Port",
                    value=mode_manager.server["address"]
                )
                tgwui_refresh_settings_gr = gr.Button("Refresh settings & voices")
    
        # Refresh settings button (only shown in local mode)
        if mode_manager.is_local:
            tgwui_refresh_settings_gr = gr.Button("Refresh settings & voices")

        # Control buttons
        with gr.Row():
            tgwui_convert_gr = gr.Button("Remove old TTS audio and leave only message texts")
            tgwui_convert_cancel_gr = gr.Button("Cancel", visible=False)
            tgwui_convert_confirm_gr = gr.Button(
                "Confirm (cannot be undone)",
                variant="stop",
                visible=False
            )
            tgwui_stop_generation_gr = gr.Button("Stop current TTS generation")
            
    # Convert history with confirmation
        convert_arr = [tgwui_convert_confirm_gr, tgwui_convert_gr, tgwui_convert_cancel_gr]
        
        tgwui_convert_gr.click(
            lambda: [
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True)
            ],
            None,
            convert_arr
        )

        tgwui_convert_confirm_gr.click(
            lambda: [
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False)
            ],
            None,
            convert_arr
        ).then(
            remove_tts_from_history,
            gradio("history"),
            gradio("history")
        ).then(
            chat.save_history,
            gradio("history", "unique_id", "character_menu", "mode"),
            None
        ).then(
            chat.redraw_html,
            gradio(ui_chat.reload_arr),
            gradio("display")
        )

        tgwui_convert_cancel_gr.click(
            lambda: [
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False)
            ],
            None,
            convert_arr
        )

        # Toggle message text in history
        tgwui_show_text_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_show_text": x}),
            tgwui_show_text_gr,
            None
        ).then(
            toggle_text_in_history,
            gradio("history"),
            gradio("history")
        ).then(
            chat.save_history,
            gradio("history", "unique_id", "character_menu", "mode"),
            None
        ).then(
            chat.redraw_html,
            gradio(ui_chat.reload_arr),
            gradio("display")
        )

        # Event functions to update the parameters in the backend
        tgwui_activate_tts_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_activate_tts": x}),
            tgwui_activate_tts_gr,
            None
        )
        
        tgwui_autoplay_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_autoplay_tts": x}),
            tgwui_autoplay_gr,
            None
        )
        
        tgwui_lowvram_enabled_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_lowvram_enabled": x}),
            tgwui_lowvram_enabled_gr,
            None
        )
        
        tgwui_lowvram_enabled_gr.change(
            lambda x: send_lowvram_request(x),
            tgwui_lowvram_enabled_gr,
            tgwui_lowvram_enabled_play_gr,
            None
        )

        # Model change handling
        tgwui_tts_dropdown_gr.change(
            tgwui_handle_ttsmodel_dropdown_change,
            tgwui_tts_dropdown_gr,
            None
        )

        # DeepSpeed settings
        tgwui_deepspeed_enabled_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_deepspeed_enabled": x}),
            tgwui_deepspeed_enabled_gr,
            None
        )
        
        tgwui_deepspeed_enabled_gr.change(
            send_deepspeed_request,
            tgwui_deepspeed_enabled_gr,
            tgwui_deepspeed_enabled_play_gr,
            None
        )

        # Voice and language settings
        tgwui_character_voice_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_character_voice": x}),
            tgwui_character_voice_gr,
            None
        )
        
        tgwui_language_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_language": x}),
            tgwui_language_gr,
            None
        )

        # TTS Settings
        tgwui_temperature_set_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_temperature_set": x}),
            tgwui_temperature_set_gr,
            None
        )
        
        tgwui_repetitionpenalty_set_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_repetitionpenalty_set": x}),
            tgwui_repetitionpenalty_set_gr,
            None
        )
        
        tgwui_generationspeed_set_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_generationspeed_set": x}),
            tgwui_generationspeed_set_gr,
            None
        )
        
        tgwui_pitch_set_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_pitch_set": x}),
            tgwui_pitch_set_gr,
            None
        )

        # Narrator settings
        tgwui_narrator_enabled_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_narrator_enabled": x}),
            tgwui_narrator_enabled_gr,
            None
        )
        
        tgwui_non_quoted_text_is_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_non_quoted_text_is": x}),
            tgwui_non_quoted_text_is_gr,
            None
        )
        
        tgwui_narrator_voice_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_narrator_voice": x}),
            tgwui_narrator_voice_gr,
            None
        )

        # RVC selection actions
        tgwui_rvc_char_voice_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_rvc_char_voice": x}),
            tgwui_rvc_char_voice_gr,
            None
        )
        
        tgwui_rvc_narr_voice_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_rvc_narr_voice": x}),
            tgwui_rvc_narr_voice_gr,
            None
        )
        tgwui_rvc_char_pitch_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_rvc_char_pitch": x}),
            tgwui_rvc_char_pitch_gr,
            None
        )
        
        tgwui_rvc_narr_pitch_gr.change(
            lambda x: mode_manager.config["tgwui"].update({"tgwui_rvc_narr_pitch": x}),
            tgwui_rvc_narr_pitch_gr,
            None
        )
                

        # Preview functionality      
        tgwui_preview_play_gr.click(
            voice_preview,
            tgwui_preview_text_gr,
            tgwui_preview_audio_gr
        )

        # Stop generation button
        tgwui_stop_generation_gr.click(
            stop_generate_tts,
            None,
            None
        )

        if not mode_manager.is_local:
            # Remote-only handlers
            tgwui_protocol_gr.change(
                tgwui_update_alltalk_protocol,
                tgwui_protocol_gr,
                None
            )
            
            tgwui_ip_address_port_gr.change(
                tgwui_update_alltalk_ip_port,
                tgwui_ip_address_port_gr,
                None
            )
            
        tgwui_refresh_settings_gr.click(
            tgwui_update_dropdowns,
            None,
            [
                tgwui_lowvram_enabled_gr,
                tgwui_deepspeed_enabled_gr,
                tgwui_character_voice_gr,
                tgwui_rvc_char_voice_gr,
                tgwui_rvc_char_pitch_gr,
                tgwui_narrator_voice_gr,
                tgwui_rvc_narr_voice_gr,
                tgwui_rvc_narr_pitch_gr,
                tgwui_tts_dropdown_gr,
                tgwui_temperature_set_gr,
                tgwui_repetitionpenalty_set_gr,
                tgwui_language_gr,
                tgwui_generationspeed_set_gr,
                tgwui_pitch_set_gr,
                tgwui_non_quoted_text_is_gr
            ]
        )       
    
    # Convert history with confirmation
    convert_arr = [tgwui_convert_confirm_gr, tgwui_convert_gr, tgwui_convert_cancel_gr]
    tgwui_convert_gr.click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True),], None, convert_arr,)
    tgwui_convert_confirm_gr.click(lambda: [ gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),], None, convert_arr,
                          ).then(remove_tts_from_history, gradio("history"), gradio("history")
                                 ).then(chat.save_history, gradio("history", "unique_id", "character_menu", "mode"), None,
                                        ).then(chat.redraw_html, gradio(ui_chat.reload_arr), gradio("display"))
    tgwui_convert_cancel_gr.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),], None, convert_arr,)

    # Toggle message text in history
    tgwui_show_text_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_show_text": x}), tgwui_show_text_gr, None
                     ).then(toggle_text_in_history, gradio("history"), gradio("history")
                            ).then(chat.save_history, gradio("history", "unique_id", "character_menu", "mode"), None,
                                   ).then(chat.redraw_html, gradio(ui_chat.reload_arr), gradio("display"))

    # Event functions to update the parameters in the backend
    tgwui_activate_tts_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_activate_tts": x}), tgwui_activate_tts_gr, None)
    tgwui_autoplay_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_autoplay_tts": x}), tgwui_autoplay_gr, None)
    tgwui_lowvram_enabled_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_lowvram_enabled": x}), tgwui_lowvram_enabled_gr, None)
    tgwui_lowvram_enabled_gr.change(lambda x: send_lowvram_request(x), tgwui_lowvram_enabled_gr, tgwui_lowvram_enabled_play_gr, None)

    # Trigger the send_reload_request function when the dropdown value changes
    tgwui_tts_dropdown_gr.change(tgwui_handle_ttsmodel_dropdown_change, tgwui_tts_dropdown_gr, None)

    tgwui_deepspeed_enabled_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_deepspeed_enabled": x}), tgwui_deepspeed_enabled_gr, None)
    tgwui_deepspeed_enabled_gr.change(send_deepspeed_request, tgwui_deepspeed_enabled_gr, tgwui_deepspeed_enabled_play_gr, None)
    tgwui_character_voice_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_character_voice": x}), tgwui_character_voice_gr, None)
    tgwui_language_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_language": x}), tgwui_language_gr, None)

    # TSS Settings
    tgwui_temperature_set_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_temperature_set": x}), tgwui_temperature_set_gr, None)
    tgwui_repetitionpenalty_set_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_repetitionpenalty_set": x}), tgwui_repetitionpenalty_set_gr, None)
    tgwui_generationspeed_set_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_generationspeed_set": x}), tgwui_generationspeed_set_gr, None)
    tgwui_pitch_set_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_pitch_set": x}), tgwui_pitch_set_gr, None)

    # Narrator selection actions
    tgwui_narrator_enabled_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_narrator_enabled": x}), tgwui_narrator_enabled_gr, None)
    tgwui_non_quoted_text_is_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_non_quoted_text_is": x}), tgwui_non_quoted_text_is_gr, None)
    tgwui_narrator_voice_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_narrator_voice": x}), tgwui_narrator_voice_gr, None)
    
    # RVC selection actions
    tgwui_rvc_char_voice_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_rvc_char_voice": x}), tgwui_rvc_char_voice_gr, None)
    tgwui_rvc_char_pitch_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_rvc_char_pitch": x}), tgwui_rvc_char_pitch_gr, None)    
    tgwui_rvc_narr_voice_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_rvc_narr_voice": x}), tgwui_rvc_narr_voice_gr, None)
    tgwui_rvc_narr_pitch_gr.change(lambda x: mode_manager.config["tgwui"].update({"tgwui_rvc_narr_pitch": x}), tgwui_rvc_narr_pitch_gr, None)