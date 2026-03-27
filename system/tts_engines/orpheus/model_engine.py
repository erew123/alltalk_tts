###############################################
# DONT CHANGE # These are base imports needed #
###############################################
import os
import sys
import json
import time
import torch
import logging
import platform
import wave
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

#############################################################################################################
#############################################################################################################
# CHANGE ME # Run any specifc imports, requirements or setup any global vaiables needed for this TTS Engine #
#############################################################################################################
#############################################################################################################

import subprocess
import numpy as np
import torchaudio

# Detect platform and import appropriate backend
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
ORPHEUS_CPP_AVAILABLE = False
ORPHEUS_VLLM_AVAILABLE = False
SNAC_AVAILABLE = False
HF_LOGGED_IN = False
ORPHEUS_INSTALL_INSTRUCTIONS = ""

def _check_huggingface_auth():
    """Check if user is logged into HuggingFace"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Try to get user info - will fail if not logged in
        user_info = api.whoami()
        return True, user_info.get('name', 'Unknown')
    except Exception:
        return False, None

def _check_huggingface_model_access(model_id="canopylabs/orpheus-3b-0.1-ft"):
    """Check if user has accepted the model license"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Try to get model info - will fail if gated and not accepted
        model_info = api.model_info(model_id)
        return True
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            return False
        # Other errors might be network issues, assume OK
        return True

def _get_install_instructions():
    """Get platform-specific installation instructions"""
    if IS_MAC:
        return (
            "Orpheus requires the following packages:\n"
            "    pip install orpheus-cpp\n"
            "    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal\n"
            "\n"
            "You also need a HuggingFace account:\n"
            "    1. Create account at: https://huggingface.co/join\n"
            "    2. Accept the model license at: https://huggingface.co/canopylabs/orpheus-3b-0.1-ft\n"
            "    3. Create access token at: https://huggingface.co/settings/tokens\n"
            "    4. Run: pip install huggingface_hub && huggingface-cli login"
        )
    else:
        return (
            "Orpheus requires the following packages:\n"
            "    pip install orpheus-speech\n"
            "    pip install vllm==0.7.3\n"
            "\n"
            "You also need a HuggingFace account:\n"
            "    1. Create account at: https://huggingface.co/join\n"
            "    2. Accept the model license at: https://huggingface.co/canopylabs/orpheus-3b-0.1-ft\n"
            "    3. Create access token at: https://huggingface.co/settings/tokens\n"
            "    4. Run: pip install huggingface_hub && huggingface-cli login"
        )

ORPHEUS_INSTALL_INSTRUCTIONS = _get_install_instructions()

def _install_orpheus_mac():
    """Install orpheus-cpp package for Mac"""
    try:
        print("#################################################################")
        print("[Orpheus] Installing required packages for Mac... This may take a while.")
        print("#################################################################")

        # Install orpheus-cpp
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "orpheus-cpp"
        ])

        # Install llama-cpp-python with Metal support
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "llama-cpp-python",
            "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/metal"
        ])

        print("##############################################################")
        print("[Orpheus] Packages installed successfully! Restarting application...")
        print("##############################################################")

        os.execv(sys.executable, ['python'] + sys.argv)

    except subprocess.CalledProcessError as e:
        print(f"[Orpheus] \033[91mError\033[0m: Failed to install packages: {str(e)}")
        raise ImportError("Could not install Orpheus packages for Mac")

def _install_orpheus_cuda():
    """Install orpheus-speech package for Windows/Linux with CUDA"""
    try:
        print("#################################################################")
        print("[Orpheus] Installing required packages for CUDA... This may take a while.")
        print("#################################################################")

        # Install orpheus-speech
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "orpheus-speech"
        ])

        # Install vLLM (pinned version)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "vllm==0.7.3"
        ])

        print("##############################################################")
        print("[Orpheus] Packages installed successfully! Restarting application...")
        print("##############################################################")

        os.execv(sys.executable, ['python'] + sys.argv)

    except subprocess.CalledProcessError as e:
        print(f"[Orpheus] \033[91mError\033[0m: Failed to install packages: {str(e)}")
        raise ImportError("Could not install Orpheus packages for CUDA")

# Check HuggingFace login status
HF_LOGGED_IN, hf_user = _check_huggingface_auth()
if HF_LOGGED_IN:
    print(f"[Orpheus] HuggingFace authenticated as: {hf_user}")
else:
    print("[Orpheus] \033[93mWarning\033[0m: Not logged into HuggingFace. Run: huggingface-cli login")

# Try to import SNAC for voice cloning
try:
    from snac import SNAC
    SNAC_AVAILABLE = True
except ImportError:
    pass

if IS_MAC:
    # Try to import orpheus-cpp for Mac/Metal
    try:
        from orpheus_cpp import OrpheusCpp
        ORPHEUS_CPP_AVAILABLE = True
        print("[Orpheus] orpheus-cpp backend available (Metal)")
    except ImportError:
        print("[Orpheus] \033[93mWarning\033[0m: orpheus-cpp not installed. Will attempt auto-install on first use.")
else:
    # Try to import orpheus-speech for CUDA
    try:
        from orpheus_tts import OrpheusModel
        ORPHEUS_VLLM_AVAILABLE = True
        print("[Orpheus] orpheus-speech backend available (vLLM/CUDA)")
    except ImportError:
        print("[Orpheus] \033[93mWarning\033[0m: orpheus-speech not installed. Will attempt auto-install on first use.")

# Built-in Orpheus voices
BUILTIN_VOICES = ["tara", "leah", "jess", "mia", "zoe", "leo", "dan", "zac"]


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

        # Orpheus-specific: Track which backend is being used
        self.backend = "cpp" if IS_MAC else "vllm"
        self.current_hf_model = None
        self.snac_model = None  # For voice cloning

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
        # Print Orpheus backend info
        if IS_MAC:
            backend_status = "Available" if ORPHEUS_CPP_AVAILABLE else "Not installed"
            print(f"[{self.branding}ENG] \033[92mOrpheus Backend   :\033[93m Metal (orpheus-cpp: {backend_status})\033[0m")
        else:
            backend_status = "Available" if ORPHEUS_VLLM_AVAILABLE else "Not installed"
            print(f"[{self.branding}ENG] \033[92mOrpheus Backend   :\033[93m CUDA (orpheus-speech: {backend_status})\033[0m")
        snac_status = "Available" if SNAC_AVAILABLE else "Not installed"
        print(f"[{self.branding}ENG] \033[92mSNAC (Cloning)    :\033[93m {snac_status}\033[0m")
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
        # Orpheus doesn't support low VRAM mode in the same way
        # Clear model from memory if needed
        if self.lowvram_enabled:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ########################################
    ########################################
    # CHANGE ME #  DeepSpeed model loading #
    ########################################
    ########################################
    async def handle_deepspeed_change(self, value):
        # Orpheus does not support DeepSpeed
        return value

    #####################################################################################
    #####################################################################################
    # CHANGE ME # scan for available models/voices that are relevant to this TTS engine #
    #####################################################################################
    #####################################################################################
    def scan_models_folder(self):
        # Orpheus has two models available from HuggingFace
        self.available_models = {
            'orpheus - orpheus-3b-0.1-ft': 'orpheus',
            'orpheus - orpheus-3b-0.1-pretrained': 'orpheus'
        }
        return self.available_models

    #############################################################
    #############################################################
    # CHANGE ME #  POPULATE FILES LIST FROM VOICES DIRECTORY ####
    #############################################################
    #############################################################
    def voices_file_list(self):
        """List available voices - built-in voices plus custom wav+reference.txt pairs"""
        try:
            voices = []
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            # ↑↑↑ Keep everything above this line ↑↑↑
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            # Add built-in Orpheus voices
            voices.extend(BUILTIN_VOICES)

            # Check for custom voice cloning voices in the voices directory
            # These require .wav files with matching .reference.txt files
            voices_dir = self.main_dir / "voices"
            if voices_dir.exists():
                def has_reference_text(wav_path):
                    """Check if a wav file has a corresponding reference text file"""
                    text_path = wav_path.with_suffix('.reference.txt')
                    return text_path.exists()

                # Add .wav files in the main "voices" directory (only if they have matching .reference.txt)
                for f in voices_dir.glob("*.wav"):
                    if has_reference_text(f):
                        # Mark custom voices with [clone] prefix for clarity
                        voices.append(f"[clone] {f.stem}")
                    else:
                        print(f"[{self.branding}ENG] Note: {f.name} needs a .reference.txt file for voice cloning") if self.debug_tts else None

                # Walk through subfolders and add subfolder names if they contain valid wav+reference.txt pairs
                for folder in voices_dir.iterdir():
                    if folder.is_dir():
                        for wav_file in folder.glob("*.wav"):
                            if has_reference_text(wav_file):
                                voices.append(f"[clone] {folder.name}/")
                                break

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

    def _is_clone_voice(self, voice):
        """Check if voice is a custom clone voice"""
        return voice.startswith("[clone] ")

    def _get_clone_voice_path(self, voice):
        """Get the path to a clone voice's wav file"""
        voice_name = voice.replace("[clone] ", "")
        voices_dir = self.main_dir / "voices"

        if voice_name.endswith("/"):
            # It's a folder, find first valid wav+reference.txt pair
            folder = voices_dir / voice_name.rstrip("/")
            for wav_file in folder.glob("*.wav"):
                if wav_file.with_suffix('.reference.txt').exists():
                    return wav_file
        else:
            # Direct wav file
            return voices_dir / f"{voice_name}.wav"
        return None

    def _get_hf_model_name(self, model_name):
        """Convert model display name to HuggingFace model ID"""
        model_map = {
            'orpheus-3b-0.1-ft': 'canopylabs/orpheus-3b-0.1-ft',
            'orpheus-3b-0.1-pretrained': 'canopylabs/orpheus-3b-0.1-pretrained',
        }
        return model_map.get(model_name, 'canopylabs/orpheus-3b-0.1-ft')

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

        # Get the HuggingFace model name
        hf_model_name = self._get_hf_model_name(model_name)
        self.current_hf_model = hf_model_name

        # Check HuggingFace authentication first
        if not HF_LOGGED_IN:
            error_msg = (
                "Orpheus requires HuggingFace authentication.\n"
                "The model is gated and requires you to accept the license.\n\n"
                "Please follow these steps:\n"
                "1. Create account at: https://huggingface.co/join\n"
                "2. Accept the model license at: https://huggingface.co/canopylabs/orpheus-3b-0.1-ft\n"
                "3. Create access token at: https://huggingface.co/settings/tokens\n"
                "4. Run: huggingface-cli login\n"
                "5. Restart AllTalk"
            )
            print(f"[{self.branding}ENG] \033[91mError\033[0m: {error_msg}")
            raise HTTPException(status_code=401, detail=error_msg)

        # Check if user has accepted the model license
        if not _check_huggingface_model_access(hf_model_name):
            error_msg = (
                f"You need to accept the license for {hf_model_name}\n"
                f"Please visit: https://huggingface.co/{hf_model_name}\n"
                "Click 'Agree and access repository', then restart AllTalk."
            )
            print(f"[{self.branding}ENG] \033[91mError\033[0m: {error_msg}")
            raise HTTPException(status_code=403, detail=error_msg)

        try:
            if IS_MAC:
                # Mac with Metal - use orpheus-cpp
                global ORPHEUS_CPP_AVAILABLE
                if not ORPHEUS_CPP_AVAILABLE:
                    print(f"[{self.branding}ENG] orpheus-cpp not installed. Attempting auto-install...")
                    _install_orpheus_mac()
                    # If we get here, installation failed (otherwise script would have restarted)
                    raise HTTPException(status_code=500, detail=f"Failed to install orpheus-cpp.\n{ORPHEUS_INSTALL_INSTRUCTIONS}")

                print(f"[{self.branding}ENG] \033[94mInitializing Orpheus TTS (Metal backend)\033[0m")
                # OrpheusCpp handles model loading internally
                # n_gpu_layers=-1 offloads all layers to Metal GPU
                # verbose=True shows tokens/second and other llama.cpp stats
                self.model = OrpheusCpp(n_gpu_layers=-1, verbose=True, lang="en")
                print(f"[{self.branding}ENG] \033[92mOrpheus initialized successfully (Metal)\033[0m")

            else:
                # Windows/Linux with CUDA - use orpheus-speech (vLLM)
                global ORPHEUS_VLLM_AVAILABLE
                if not ORPHEUS_VLLM_AVAILABLE:
                    print(f"[{self.branding}ENG] orpheus-speech not installed. Attempting auto-install...")
                    _install_orpheus_cuda()
                    # If we get here, installation failed (otherwise script would have restarted)
                    raise HTTPException(status_code=500, detail=f"Failed to install orpheus-speech.\n{ORPHEUS_INSTALL_INSTRUCTIONS}")

                print(f"[{self.branding}ENG] \033[94mInitializing Orpheus TTS (vLLM backend): {hf_model_name}\033[0m")
                self.model = OrpheusModel(
                    model_name=hf_model_name,
                    max_model_len=2048
                )
                print(f"[{self.branding}ENG] \033[92mOrpheus initialized successfully (vLLM)\033[0m")

        except HTTPException:
            raise
        except Exception as e:
            error_str = str(e)
            # Check for common HuggingFace auth errors
            if "401" in error_str or "403" in error_str or "gated" in error_str.lower():
                error_msg = (
                    f"HuggingFace authentication error: {error_str}\n\n"
                    "Please ensure you have:\n"
                    "1. Accepted the model license at: https://huggingface.co/canopylabs/orpheus-3b-0.1-ft\n"
                    "2. Logged in with: huggingface-cli login"
                )
                print(f"[{self.branding}ENG] \033[91mError\033[0m: {error_msg}")
                raise HTTPException(status_code=401, detail=error_msg)
            print(f"[{self.branding}ENG] \033[91mError initializing Orpheus:\033[0m {error_str}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Orpheus: {error_str}")

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

        # Clear model reference
        self.model = None
        self.current_hf_model = None
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
        if tts_method.startswith("orpheus"):
            model_name = tts_method.split(" - ")[1] if " - " in tts_method else "orpheus-3b-0.1-ft"
            backend_name = "Metal" if IS_MAC else "vLLM/CUDA"
            print(f"[{self.branding}ENG]\033[94m Model/Engine :\033[93m {model_name}\033[94m loading with\033[93m {backend_name}\033[0m")
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
            print(f"[{self.branding}GEN] \033[94mVoice:\033[93m {voice} \033[94mTemp:\033[93m {temperature} \033[94mRep Penalty:\033[93m {repetition_penalty}\033[0m") if self.debug_tts else None

            # Check if this is a clone voice
            if self._is_clone_voice(voice):
                # Voice cloning requires pretrained model and SNAC
                # Currently not fully implemented - show helpful message
                print(f"[{self.branding}ENG] \033[93mVoice cloning requested for:\033[0m {voice}")

                if not SNAC_AVAILABLE:
                    raise HTTPException(status_code=400, detail="Voice cloning requires SNAC. Install with: pip install snac")

                # Get the reference audio path
                ref_audio_path = self._get_clone_voice_path(voice)
                if ref_audio_path is None or not ref_audio_path.exists():
                    raise HTTPException(status_code=400, detail=f"Reference audio not found for voice: {voice}")

                ref_text_path = ref_audio_path.with_suffix('.reference.txt')
                if not ref_text_path.exists():
                    raise HTTPException(status_code=400, detail=f"Reference text not found: {ref_text_path.name}")

                # For now, voice cloning with custom voices requires the pretrained model
                # and full transformers setup - show informative message
                print(f"[{self.branding}ENG] \033[91mNote:\033[0m Voice cloning with custom voices requires the pretrained model.")
                print(f"[{self.branding}ENG] Reference audio: {ref_audio_path}")
                print(f"[{self.branding}ENG] Reference text: {ref_text_path}")

                # TODO: Implement full voice cloning pipeline with SNAC encoding
                # For now, fall back to the default 'tara' voice with a warning
                print(f"[{self.branding}ENG] \033[93mFalling back to 'tara' voice. Full voice cloning coming soon.\033[0m")
                voice = "tara"

            if IS_MAC:
                # Mac with Metal - use orpheus-cpp
                if streaming:
                    # Streaming mode - yield chunks directly
                    chunk_count = 0
                    for i, (sr, chunk) in enumerate(self.model.stream_tts_sync(text, options={"voice_id": voice})):
                        # Process chunk for streaming
                        if len(chunk.shape) > 1:
                            chunk = chunk.flatten()
                        if chunk.dtype == np.float32 or chunk.dtype == np.float64:
                            chunk = (chunk * 32767).astype(np.int16)
                        yield chunk.tobytes()
                        chunk_count += 1
                    if chunk_count == 0:
                        print(f"[{self.branding}ENG] \033[91mError\033[0m: No audio generated during streaming")
                        raise HTTPException(status_code=500, detail="No audio generated")
                else:
                    # Non-streaming mode - collect and write to file
                    audio_chunks = []
                    for i, (sr, chunk) in enumerate(self.model.stream_tts_sync(text, options={"voice_id": voice})):
                        audio_chunks.append(chunk)

                    if audio_chunks:
                        # Concatenate all chunks
                        full_audio = np.concatenate(audio_chunks, axis=1)
                        # Flatten to 1D if needed
                        if len(full_audio.shape) > 1:
                            full_audio = full_audio.flatten()
                        # Convert to int16 for WAV
                        if full_audio.dtype == np.float32 or full_audio.dtype == np.float64:
                            full_audio = (full_audio * 32767).astype(np.int16)
                        # Write WAV file at 24kHz
                        with wave.open(str(output_file), "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(24000)
                            wf.writeframes(full_audio.tobytes())
                    else:
                        print(f"[{self.branding}ENG] \033[91mError\033[0m: No audio generated")
                        raise HTTPException(status_code=500, detail="No audio generated")

            else:
                # Windows/Linux with vLLM
                audio_stream = self.model.generate_speech(
                    prompt=text,
                    voice=voice,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty
                )

                if streaming:
                    # Streaming mode - yield chunks directly
                    for chunk in audio_stream:
                        yield chunk
                else:
                    # Non-streaming mode - write to file
                    with wave.open(str(output_file), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(24000)
                        for chunk in audio_stream:
                            wf.writeframes(chunk)

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
