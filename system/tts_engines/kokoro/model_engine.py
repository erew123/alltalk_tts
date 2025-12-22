###############################################
# DONT CHANGE # These are base imports needed #
###############################################
import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from fastapi import (HTTPException)
logging.disable(logging.WARNING)
#################################################################
# DONT CHANGE # Get Pytorch & Python versions & setup DeepSpeed #
#################################################################
pytorch_version = torch.__version__
cuda_version = torch.version.cuda
major, minor, micro = sys.version_info[:3]
python_version = f"{major}.{minor}.{micro}"
try:
    import deepspeed
    deepspeed_available = True
except ImportError:
    deepspeed_available = False
    pass

#############################################################################################################
#############################################################################################################
# CHANGE ME # Run any specifc imports, requirements or setup any global vaiables needed for this TTS Engine #
#############################################################################################################
#############################################################################################################

import subprocess
import platform

# Track availability status
KOKORO_AVAILABLE = False
ESPEAK_AVAILABLE = False
ESPEAK_INSTALL_INSTRUCTIONS = ""

def _check_espeak_installed():
    """Check if espeak-ng is installed and return (available, library_path, data_path, install_instructions)"""
    system = platform.system()

    if system == "Darwin":  # macOS
        # Check Homebrew paths (Apple Silicon first, then Intel)
        paths_to_check = [
            ("/opt/homebrew/lib/libespeak-ng.dylib", "/opt/homebrew/share/espeak-ng-data"),  # Apple Silicon
            ("/usr/local/lib/libespeak-ng.dylib", "/usr/local/share/espeak-ng-data"),  # Intel
        ]
        for lib_path, data_path in paths_to_check:
            if os.path.exists(lib_path):
                return True, lib_path, data_path, ""

        return False, None, None, (
            "espeak-ng is not installed. Please install it:\n"
            "    brew install espeak-ng\n"
            "Then restart AllTalk."
        )

    elif system == "Linux":
        # Check common Linux paths
        lib_paths = [
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",  # Debian/Ubuntu x64
            "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",  # Debian/Ubuntu ARM64
            "/usr/lib/libespeak-ng.so.1",  # Arch/Fedora
            "/usr/lib64/libespeak-ng.so.1",  # Some Fedora/RHEL
        ]
        data_paths = [
            "/usr/share/espeak-ng-data",
            "/usr/lib/espeak-ng-data",
        ]

        found_lib = None
        found_data = None
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                found_lib = lib_path
                break
        for data_path in data_paths:
            if os.path.exists(data_path):
                found_data = data_path
                break

        if found_lib:
            return True, found_lib, found_data, ""

        return False, None, None, (
            "espeak-ng is not installed. Please install it:\n"
            "    Ubuntu/Debian: sudo apt-get install espeak-ng\n"
            "    Fedora/RHEL:   sudo dnf install espeak-ng\n"
            "    Arch:          sudo pacman -S espeak-ng\n"
            "Then restart AllTalk."
        )

    elif system == "Windows":
        # Check if espeak-ng is in PATH
        try:
            result = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return True, None, None, ""  # Windows uses PATH, not direct library paths
        except FileNotFoundError:
            pass

        # Check common Windows install locations
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        paths_to_check = [
            os.path.join(program_files, "eSpeak NG"),
            os.path.join(program_files_x86, "eSpeak NG"),
        ]
        for path in paths_to_check:
            if os.path.exists(path):
                lib_path = os.path.join(path, "libespeak-ng.dll")
                data_path = os.path.join(path, "espeak-ng-data")
                if os.path.exists(lib_path):
                    return True, lib_path, data_path, ""

        return False, None, None, (
            "espeak-ng is not installed. Please install it:\n"
            "    1. Download from: https://github.com/espeak-ng/espeak-ng/releases\n"
            "    2. Run the .msi installer\n"
            "    3. Add espeak-ng to your system PATH\n"
            "    4. Restart AllTalk"
        )

    return False, None, None, "Unknown operating system"

def _setup_espeak_env():
    """Set up espeak-ng environment variables for Kokoro/Misaki"""
    global ESPEAK_AVAILABLE, ESPEAK_INSTALL_INSTRUCTIONS

    available, lib_path, data_path, instructions = _check_espeak_installed()
    ESPEAK_AVAILABLE = available
    ESPEAK_INSTALL_INSTRUCTIONS = instructions

    if available:
        if lib_path:
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib_path
        if data_path:
            os.environ["ESPEAK_DATA_PATH"] = data_path
        print("[Kokoro] espeak-ng found and configured")
    else:
        print(f"[Kokoro] \033[93mWarning\033[0m: {instructions}")

_setup_espeak_env()

def _install_kokoro():
    """Install kokoro package if not present"""
    try:
        print("#################################################################")
        print("[Kokoro] Installing required packages... This may take a while.")
        print("#################################################################")

        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "kokoro>=0.9.2", "soundfile", "phonemizer"
        ])

        print("##############################################################")
        print("[Kokoro] Packages installed successfully! Restarting application...")
        print("##############################################################")

        # Restart the current script
        os.execv(sys.executable, ['python'] + sys.argv)

    except subprocess.CalledProcessError as e:
        print(f"[Kokoro] \033[91mError\033[0m: Failed to install packages: {str(e)}")
        raise ImportError("Could not install Kokoro packages")

# Patch the EspeakWrapper if needed (fixes set_data_path AttributeError)
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    if not hasattr(EspeakWrapper, 'set_data_path') and hasattr(EspeakWrapper, 'data_path'):
        EspeakWrapper.set_data_path = classmethod(lambda cls, path: setattr(cls, '_data_path', path))
    if not hasattr(EspeakWrapper, 'set_library') and hasattr(EspeakWrapper, 'library'):
        EspeakWrapper.set_library = classmethod(lambda cls, path: setattr(cls, '_library', path))
except ImportError:
    pass

# Try to import Kokoro
try:
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("[Kokoro] Kokoro package not found. Will attempt to install on first use.")
    # Import numpy anyway as it's likely available
    try:
        import numpy as np
    except ImportError:
        pass

# Language code mapping for Kokoro
# Maps AllTalk language codes to Kokoro's internal codes
LANG_CODES = {
    'en': 'a',       # American English (default)
    'en-us': 'a',    # American English
    'en-gb': 'b',    # British English
    'ja': 'j',       # Japanese
    'zh': 'z',       # Mandarin Chinese
    'es': 'e',       # Spanish
    'fr': 'f',       # French
    'hi': 'h',       # Hindi
    'it': 'i',       # Italian
    'pt': 'p',       # Brazilian Portuguese
}

# Voice to language mapping (first letter of voice code indicates language)
VOICE_LANG_MAP = {
    'a': 'a',  # American English
    'b': 'b',  # British English
    'j': 'j',  # Japanese
    'z': 'z',  # Chinese
    'e': 'e',  # Spanish
    'f': 'f',  # French
    'h': 'h',  # Hindi
    'i': 'i',  # Italian
    'p': 'p',  # Portuguese
}


#################################################################################################################################
# DONT CHANGE # Do not change the Class name from tts_class as this is what will be imported into the main tts_server.py script #
#################################################################################################################################
class tts_class:
    def __init__(self):
        ########################################################################
        # DONT CHANGE # Sets up the base variables required for any tts engine #
        ########################################################################
        self.branding = None
        self.this_dir = Path(__file__).parent.resolve()
        self.main_dir = Path(__file__).parent.parent.parent.parent.resolve()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda_is_available = torch.cuda.is_available()
        self.tts_generating_lock = False
        self.tts_stop_generation = False
        self.tts_narrator_generatingtts = False
        self.model = None
        self.is_tts_model_loaded = False
        self.current_model_loaded = None
        self.available_models = None
        self.setup_has_run = False
        ##############################################################################################
        # DONT CHANGE # Load in a list of the available TTS engines and the currently set TTS engine #
        ##############################################################################################
        tts_engines_file = os.path.join(self.main_dir, "system", "tts_engines", "tts_engines.json")
        with open(tts_engines_file, "r") as f:
            tts_engines_data = json.load(f)
        self.engines_available = [engine["name"] for engine in tts_engines_data["engines_available"]]
        self.engine_loaded = tts_engines_data["engine_loaded"]
        self.selected_model = tts_engines_data["selected_model"]
        ############################################################################
        # DONT CHANGE # Pull out all the settings for the currently set TTS engine #
        ############################################################################
        with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
            tts_model_loaded = json.load(f)
        # Access the model details
        self.manufacturer_name = tts_model_loaded["model_details"]["manufacturer_name"]
        self.manufacturer_website = tts_model_loaded["model_details"]["manufacturer_website"]
        # Access the features the model is capable of:
        self.audio_format = tts_model_loaded["model_capabilties"]["audio_format"]
        self.deepspeed_capable = tts_model_loaded["model_capabilties"]["deepspeed_capable"]
        self.deepspeed_available = 'deepspeed' in globals()
        self.generationspeed_capable = tts_model_loaded["model_capabilties"]["generationspeed_capable"]
        self.languages_capable = tts_model_loaded["model_capabilties"]["languages_capable"]
        self.lowvram_capable = tts_model_loaded["model_capabilties"]["lowvram_capable"]
        self.multimodel_capable = tts_model_loaded["model_capabilties"]["multimodel_capable"]
        self.repetitionpenalty_capable = tts_model_loaded["model_capabilties"]["repetitionpenalty_capable"]
        self.streaming_capable = tts_model_loaded["model_capabilties"]["streaming_capable"]
        self.temperature_capable = tts_model_loaded["model_capabilties"]["temperature_capable"]
        self.multivoice_capable = tts_model_loaded["model_capabilties"]["multivoice_capable"]
        self.pitch_capable = tts_model_loaded["model_capabilties"]["pitch_capable"]
        # Access the current engine settings
        self.def_character_voice = tts_model_loaded["settings"]["def_character_voice"]
        self.def_narrator_voice = tts_model_loaded["settings"]["def_narrator_voice"]
        self.deepspeed_enabled = tts_model_loaded["settings"]["deepspeed_enabled"]
        self.engine_installed = tts_model_loaded["settings"]["engine_installed"]
        self.generationspeed_set = tts_model_loaded["settings"]["generationspeed_set"]
        self.lowvram_enabled = tts_model_loaded["settings"]["lowvram_enabled"]
        # Check if someone has enabled lowvram on a system that's not CUDA enabled
        self.lowvram_enabled = False if not torch.cuda.is_available() else self.lowvram_enabled
        self.repetitionpenalty_set = tts_model_loaded["settings"]["repetitionpenalty_set"]
        self.temperature_set = tts_model_loaded["settings"]["temperature_set"]
        self.pitch_set = tts_model_loaded["settings"]["pitch_set"]
        # Gather the OpenAI API Voice Mappings
        self.openai_alloy = tts_model_loaded["openai_voices"]["alloy"]
        self.openai_echo = tts_model_loaded["openai_voices"]["echo"]
        self.openai_fable = tts_model_loaded["openai_voices"]["fable"]
        self.openai_nova = tts_model_loaded["openai_voices"]["nova"]
        self.openai_onyx = tts_model_loaded["openai_voices"]["onyx"]
        self.openai_shimmer = tts_model_loaded["openai_voices"]["shimmer"]
        ###################################################################
        # DONT CHANGE #  Load params and api_defaults from confignew.json #
        ###################################################################
        configfile_path = self.main_dir / "confignew.json"
        with open(configfile_path, "r") as configfile:
            configfile_data = json.load(configfile)
        self.branding = configfile_data.get("branding", "")
        self.params = configfile_data
        self.debug_tts = configfile_data.get("debugging").get("debug_tts")
        self.debug_tts_variables = configfile_data.get("debugging").get("debug_tts_variables")

        # Kokoro-specific: Store pipeline instances for each language
        self.pipelines = {}
        self.current_lang_code = 'a'  # Default to American English

    ################################################################
    # DONT CHANGE #  Print out Python, CUDA, DeepSpeed versions ####
    ################################################################
    def printout_versions(self):
        if deepspeed_available:
            print(f"[{self.branding}ENG] \033[92mDeepSpeed version :\033[93m",deepspeed.__version__,"\033[0m")
        else:
            print(f"[{self.branding}ENG] \033[92mDeepSpeed version :\033[93m Not available\033[0m")
        print(f"[{self.branding}ENG] \033[92mPython Version    :\033[93m {python_version}\033[0m")
        print(f"[{self.branding}ENG] \033[92mPyTorch Version   :\033[93m {pytorch_version}\033[0m")
        if cuda_version is None:
            print(f"[{self.branding}ENG] \033[92mCUDA Version      :\033[91m Not available\033[0m")
        else:
            print(f"[{self.branding}ENG] \033[92mCUDA Version      :\033[93m {cuda_version}\033[0m")
        print(f"[{self.branding}ENG]")
        return

    ###################################################################################
    ###################################################################################
    # CHANGE ME # Inital setup of the model and engine. Called when the script starts #
    ###################################################################################
    ###################################################################################
    async def setup(self):
        self.printout_versions()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        self.available_models = self.scan_models_folder()
        if self.selected_model:
            tts_model = f"{self.selected_model}"
            if tts_model in self.available_models:
                await self.handle_tts_method_change(tts_model)
                self.current_model_loaded = tts_model

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # ↓↓↓ Keep everything below this line ↓↓↓
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            else:
                self.current_model_loaded = "No Models Available"
                print(f"[{self.branding}ENG] \033[91mError\033[0m: Selected model '{self.selected_model}' not found.")
        self.setup_has_run = True

    ##################################
    ##################################
    # CHANGE ME #  Low VRAM Swapping #
    ##################################
    ##################################
    async def handle_lowvram_change(self):
        # Kokoro pipeline handles device management internally
        # We can clear cached pipelines to free memory
        if self.lowvram_enabled:
            self.pipelines = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ########################################
    ########################################
    # CHANGE ME #  DeepSpeed model loading #
    ########################################
    ########################################
    async def handle_deepspeed_change(self, value):
        # Kokoro does not support DeepSpeed
        return value

    #####################################################################################
    #####################################################################################
    # CHANGE ME # scan for available models/voices that are relevant to this TTS engine #
    #####################################################################################
    #####################################################################################
    def scan_models_folder(self):
        # Kokoro uses the kokoro pip package which manages models internally
        # We just report a single "model" that represents the Kokoro engine
        self.available_models = {'kokoro - kokoro_v1.0': 'kokoro'}
        return self.available_models

    #############################################################
    #############################################################
    # CHANGE ME #  POPULATE FILES LIST FROM VOICES DIRECTORY ####
    #############################################################
    #############################################################
    def voices_file_list(self):
        try:
            voices = []
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            # ↑↑↑ Keep everything above this line ↑↑↑
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            voices_file = os.path.join(self.this_dir, "kokoro_voices.json")
            if os.path.exists(voices_file):
                with open(voices_file, "r") as f:
                    voices_data = json.load(f)
                    return [voice["voice_name"] for voice in voices_data["voices"]]

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # ↓↓↓ Keep everything below this line ↓↓↓
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            if not voices:
                return ["No Voices Found"]
            return voices
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Voices not found. Cannot load a list of voices.")
            print(f"[{self.branding}ENG]")
            return ["No Voices Found"]

    def _is_voice_blend(self, voice_string):
        """Check if voice string is a blend (contains : and ,)"""
        return ':' in voice_string and ',' in voice_string

    def _parse_voice_blend(self, voice_string):
        """
        Parse a voice blend string into components.
        Format: 'voice1:weight1,voice2:weight2' -> [('voice1', 60), ('voice2', 40)]
        """
        components = []
        parts = voice_string.split(',')
        for part in parts:
            part = part.strip()
            if ':' in part:
                voice, weight = part.rsplit(':', 1)
                try:
                    weight = int(weight)
                except ValueError:
                    weight = 50  # Default weight if parsing fails
                components.append((voice.strip(), weight))
            else:
                components.append((part.strip(), 50))
        return components

    def _get_voice_info(self, voice_name):
        """Get voice code and language from voice name. Supports voice blending."""
        # Check if this is a voice blend
        if self._is_voice_blend(voice_name):
            # For blends, the voice_code IS the blend string
            # Get language from the first voice in the blend
            components = self._parse_voice_blend(voice_name)
            if components:
                first_voice = components[0][0]
                # Look up the first voice to get its language
                voices_file = os.path.join(self.this_dir, "kokoro_voices.json")
                if os.path.exists(voices_file):
                    with open(voices_file, "r") as f:
                        voices_data = json.load(f)
                        for voice in voices_data["voices"]:
                            if voice["voice_name"] == first_voice or voice.get("voice_code") == first_voice:
                                return voice_name, voice.get("language", "en-us")
                # If not found in voices file, derive language from voice code
                return voice_name, "en-us"
            return voice_name, "en-us"

        # Standard voice lookup
        voices_file = os.path.join(self.this_dir, "kokoro_voices.json")
        if os.path.exists(voices_file):
            with open(voices_file, "r") as f:
                voices_data = json.load(f)
                for voice in voices_data["voices"]:
                    if voice["voice_name"] == voice_name:
                        return voice.get("voice_code", voice_name), voice.get("language", "en-us")
        # Default: assume voice_name is the voice_code
        return voice_name, "en-us"

    def _get_lang_code(self, voice_code):
        """Get Kokoro language code from voice code. Handles blended voices."""
        # If it's a blend, extract the first voice code
        if self._is_voice_blend(voice_code):
            components = self._parse_voice_blend(voice_code)
            if components:
                voice_code = components[0][0]

        if len(voice_code) >= 2:
            first_char = voice_code[0]
            return VOICE_LANG_MAP.get(first_char, 'a')
        return 'a'  # Default to American English

    def _get_pipeline(self, lang_code):
        """Get or create a Kokoro pipeline for the specified language"""
        if lang_code not in self.pipelines:
            print(f"[{self.branding}ENG] \033[94mInitializing Kokoro pipeline for language:\033[93m {lang_code}\033[0m")
            self.pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return self.pipelines[lang_code]

    #############################
    #############################
    # CHANGE ME # Model loading #
    #############################
    #############################
    async def api_manual_load_model(self, model_name):
        if model_name == "No Models Found":
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load.")
            raise HTTPException(status_code=400, detail="No models for this TTS engine were found to load.")
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # Check for espeak-ng first (required system dependency)
        if not ESPEAK_AVAILABLE:
            error_msg = f"Kokoro requires espeak-ng to be installed.\n{ESPEAK_INSTALL_INSTRUCTIONS}"
            print(f"[{self.branding}ENG] \033[91mError\033[0m: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        # Check if Kokoro is installed, auto-install if not
        global KOKORO_AVAILABLE
        if not KOKORO_AVAILABLE:
            print(f"[{self.branding}ENG] Kokoro not installed. Attempting auto-install...")
            _install_kokoro()
            # If we get here, installation failed (otherwise script would have restarted)
            raise HTTPException(status_code=500, detail="Failed to install Kokoro. Please install manually: pip install kokoro>=0.9.2 soundfile phonemizer")

        # Initialize the default pipeline (American English)
        # Other language pipelines will be created on-demand
        try:
            print(f"[{self.branding}ENG] \033[94mInitializing Kokoro TTS engine\033[0m")
            self.pipelines['a'] = KPipeline(lang_code='a')
            self.current_lang_code = 'a'
            print(f"[{self.branding}ENG] \033[92mKokoro pipeline initialized successfully\033[0m")
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError initializing Kokoro:\033[0m {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Kokoro: {str(e)}")

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.is_tts_model_loaded = True
        return None

    ###############################
    ###############################
    # CHANGE ME # Model unloading #
    ###############################
    ###############################
    async def unload_model(self):
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # Clear all cached pipelines
        self.pipelines = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.is_tts_model_loaded = False
        return None

    ###################################################################################################################################
    ###################################################################################################################################
    # CHANGE ME # Model changing. Unload out old model and load in a new one # XTTS is very unusal as it has 2x model loading methods #
    ###################################################################################################################################
    ###################################################################################################################################
    async def handle_tts_method_change(self, tts_method):
        generate_start_time = time.time()
        if "No Models Available" in self.available_models:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load. Please download a model.")
            return False
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        await self.unload_model()
        if tts_method.startswith("kokoro"):
            model_name = tts_method.split(" - ")[1] if " - " in tts_method else "kokoro_v1.0"
            print(f"[{self.branding}ENG]\033[94m Model/Engine :\033[93m {model_name}\033[94m loading into\033[93m", self.device,"\033[0m")
            await self.api_manual_load_model(model_name)
            self.current_model_loaded = tts_method
        else:
            self.current_model_loaded = None
            return False

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m")
        return True

    ##########################################################################################################################################
    ##########################################################################################################################################
    # CHANGE ME # Model changing. Unload out old model and load in a new one # XTTS is very unusal as it has 2x model TTS generation methods #
    ##########################################################################################################################################
    ##########################################################################################################################################
    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming):
        if voice == "No Voices Found":
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No voices found to generate TTS.")
            raise HTTPException(status_code=400, detail="No voices found to generate TTS.")
        if not self.is_tts_model_loaded:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: You currently have no TTS model loaded.")
            raise HTTPException(status_code=400, detail="You currently have no TTS model loaded.")
        self.tts_generating_lock = True
        generate_start_time = time.time()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        try:
            # Get voice code and determine language
            voice_code, voice_lang = self._get_voice_info(voice)
            lang_code = self._get_lang_code(voice_code)

            # Get or create pipeline for this language
            pipeline = self._get_pipeline(lang_code)

            print(f"[{self.branding}GEN] \033[94mVoice:\033[93m {voice_code} \033[94mLanguage:\033[93m {lang_code} \033[94mSpeed:\033[93m {speed}\033[0m") if self.debug_tts else None

            # Generate audio using Kokoro pipeline
            # The pipeline returns a generator of (graphemes, phonemes, audio) tuples
            audio_chunks = []
            for gs, ps, audio in pipeline(text, voice=voice_code, speed=speed):
                audio_chunks.append(audio)

            # Concatenate all audio chunks
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                # Write output file at 24kHz (Kokoro's native sample rate)
                sf.write(output_file, full_audio, 24000)
            else:
                print(f"[{self.branding}ENG] \033[91mError\033[0m: No audio generated")
                raise HTTPException(status_code=500, detail="No audio generated")

            # Fake Streaming function - Kokoro doesn't support streaming
            wavs = None
            if streaming and wavs is not None:
                for wav_chunk in wavs:
                    yield wav_chunk.numpy().tobytes()

        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError generating TTS:\033[0m {str(e)}")
            self.tts_generating_lock = False
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[{self.branding}GEN] \033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{self.lowvram_enabled} \033[94mDeepSpeed: \033[33m{self.deepspeed_enabled}\033[0m")
        if self.lowvram_enabled and self.device == "cuda" and self.tts_narrator_generatingtts == False:
            await self.handle_lowvram_change()
        self.tts_generating_lock = False
        return
