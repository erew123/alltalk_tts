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
deepspeed_available = False

#############################################################################################################
#############################################################################################################
# CHANGE ME # Run any specifc imports, requirements or setup any global vaiables needed for this TTS Engine #
#############################################################################################################
#############################################################################################################
import soundfile as sf
import tempfile
from pydub import AudioSegment, silence
import re
import torchaudio
import numpy as np
import subprocess

def install_fish_speech():
    """Install fish-speech package if not present"""
    try:
        print("#################################################################")
        print("Installing required packages for Fish Speech... This may take a while.")
        print("################################################################")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "fish-speech"
        ])

        print("##############################################################")
        print("Fish Speech installed successfully! Restarting application...")
        print("##############################################################")

        # Restart the current script
        os.execv(sys.executable, ['python'] + sys.argv)

    except subprocess.CalledProcessError as e:
        print("########################################################")
        print(f"Failed to install Fish Speech: {str(e)}")
        print("########################################################")
        raise ImportError("Could not install Fish Speech package")

# Try to import Fish Speech components
try:
    from fish_speech.models.text2semantic.inference import (
        DualARTransformer,
        generate_long,
    )
    from fish_speech.models.dac.inference import load_model as load_codec_model
    from fish_speech.text import clean_text
    from fish_speech.tokenizer import FishTokenizer
    FISH_SPEECH_AVAILABLE = True
except ImportError:
    FISH_SPEECH_AVAILABLE = False
    print("[Fish Speech] Fish Speech package not found. Will attempt to install on first use.")


#################################################################################################################################
# DONT CHANGE # Do not change the Class name from tts_class as this is what will be imported into the main tts_server.py script #
#################################################################################################################################
class tts_class:
    def __init__(self):
        ########################################################################
        # DONT CHANGE # Sets up the base variables required for any tts engine #
        ########################################################################
        self.branding = None
        self.this_dir = Path(__file__).parent.resolve()                         # Sets up self.this_dir as a variable for the folder THIS script is running in.
        self.main_dir = Path(__file__).parent.parent.parent.parent.resolve()    # Sets up self.main_dir as a variable for the folder AllTalk is running in
        # Device selection: CUDA > MPS (Metal on Mac) > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.cuda_is_available = torch.cuda.is_available()                      # Sets up cuda_is_available as a True/False to track if Nvidia/CUDA was found on the system
        self.mps_is_available = torch.backends.mps.is_available()               # Track if MPS (Metal) is available on Mac
        self.tts_generating_lock = False                                        # Used to lock and unlock the tts generation process at the start/end of tts generation.
        self.tts_stop_generation = False                                        # Used in conjunction with tts_generating_lock to call for a stop to the current generation. If called (set True) it needs to be set back to False when generation has been stopped.
        self.tts_narrator_generatingtts = False                                 # Used to track if the current tts processes is narrator based. This can be used in conjunction with lowvram and device to avoid moving model between GPU(CUDA)<>RAM(CPU) each chunk of narrated text generated.
        self.model = None                                                       # If loading a model into CUDA/VRAM/RAM "model" is used as the variable name to load and interact with (see the XTTS model_engine script for examples.)
        self.is_tts_model_loaded = False                                        # Used to track if a model is actually loaded in and error/fail things like TTS generation if its False
        self.current_model_loaded = None                                        # Stores the name of the currenly loaded in model
        self.available_models = None                                            # List of available models found by "def scan_models_folder"
        self.setup_has_run = False                                              # Tracks if async def setup(self) has run, by setting to True, so that the /api/ready endpoint can provide a "Ready" status
        ##############################################################################################
        # DONT CHANGE # Load in a list of the available TTS engines and the currently set TTS engine #
        ##############################################################################################
        tts_engines_file = os.path.join(self.main_dir, "system", "tts_engines", "tts_engines.json")
        with open(tts_engines_file, "r") as f:
            tts_engines_data = json.load(f)
        self.engines_available = [engine["name"] for engine in tts_engines_data["engines_available"]]       # A list of ALL the TTS engines available to be loaded by AllTalk
        self.engine_loaded = tts_engines_data["engine_loaded"]                                              # In "tts_engines.json" what is the currently set TTS engine loading into AllTalk
        self.selected_model = tts_engines_data["selected_model"]                                            # In "tts_engines.json" what is the currently set TTS model loading into AllTalk
        ############################################################################
        # DONT CHANGE # Pull out all the settings for the currently set TTS engine #
        ############################################################################
        with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
            tts_model_loaded = json.load(f)
        # Access the model details
        self.manufacturer_name = tts_model_loaded["model_details"]["manufacturer_name"]                     # The company/person/body that generated the TTS engine/models etc
        self.manufacturer_website = tts_model_loaded["model_details"]["manufacturer_website"]               # The website of the company/person/body where people can find more information
        # Access the features the model is capable of:
        self.audio_format = tts_model_loaded["model_capabilties"]["audio_format"]                           # This details the audio format your TTS engine is set to generate TTS in e.g. wav, mp3, flac, opus, acc, pcm. Please use only 1x format.
        self.deepspeed_capable = tts_model_loaded["model_capabilties"]["deepspeed_capable"]                 # Is your model capable of DeepSpeed
        self.deepspeed_available = 'deepspeed' in globals()                                                 # When we did the import earlier, at the top of this script, was DeepSpeed available for use
        self.generationspeed_capable = tts_model_loaded["model_capabilties"]["generationspeed_capable"]     # Does this TTS engine support changing the speed of the generated TTS
        self.languages_capable = tts_model_loaded["model_capabilties"]["languages_capable"]                 # Are the actual models themselves capable of generating in multiple languages OR is each model language specific
        self.lowvram_capable = tts_model_loaded["model_capabilties"]["lowvram_capable"]                     # Is this engine capable of using low VRAM (moving the model between CPU And GPU memory)
        self.multimodel_capable = tts_model_loaded["model_capabilties"]["multimodel_capable"]               # Is there just the one model or are there multiple models this engine supports.
        self.repetitionpenalty_capable = tts_model_loaded["model_capabilties"]["repetitionpenalty_capable"] # Is this TTS engine capable of changing the repititon penalty
        self.streaming_capable = tts_model_loaded["model_capabilties"]["streaming_capable"]                 # Is this TTS engine capabale of generating streaming audio
        self.temperature_capable = tts_model_loaded["model_capabilties"]["temperature_capable"]             # Is this TTS engine capable of changing the temperature of the models
        self.multivoice_capable = tts_model_loaded["model_capabilties"]["multivoice_capable"]               # Are the models multi-voice or single vocice models
        self.pitch_capable = tts_model_loaded["model_capabilties"]["pitch_capable"]                         # Is this TTS engine capable of changing the pitch of the genrated TTS
        # Access the current enginesettings
        self.def_character_voice = tts_model_loaded["settings"]["def_character_voice"]                      # What is the current default main/character voice that will be used if no voice specified.
        self.def_narrator_voice = tts_model_loaded["settings"]["def_narrator_voice"]                        # What is the current default narrator voice that will be used if no voice specified.
        self.deepspeed_enabled = tts_model_loaded["settings"]["deepspeed_enabled"]                          # If its available, is DeepSpeed enabled for the TTS engine
        self.engine_installed = tts_model_loaded["settings"]["engine_installed"]                            # Has the TTS engine been setup/installed (not curently used)
        self.generationspeed_set = tts_model_loaded["settings"]["generationspeed_set"]                      # What is the set/stored speed for generation.
        self.lowvram_enabled = tts_model_loaded["settings"]["lowvram_enabled"]                              # If its available, is LowVRAM enabled for the TTS engine
        # Check if someone has enabled lowvram on a system that's not CUDA enabled
        self.lowvram_enabled = False if not torch.cuda.is_available() else self.lowvram_enabled             # If LowVRAM is mistakenly set and CUDA is not available, this will force it back off
        self.repetitionpenalty_set = tts_model_loaded["settings"]["repetitionpenalty_set"]                  # What is the currenly set repitition policy of the model (If it support repetition)
        self.temperature_set = tts_model_loaded["settings"]["temperature_set"]                              # What is the currenly set temperature of the model (If it support temp)
        self.pitch_set = tts_model_loaded["settings"]["pitch_set"]                                          # What is the currenly set pitch of the model (If it support temp)
        # Gather the OpenAI API Voice Mappings
        self.openai_alloy = tts_model_loaded["openai_voices"]["alloy"]                                      # The TTS engine voice that will be mapped to Open AI Alloy voice
        self.openai_echo = tts_model_loaded["openai_voices"]["echo"]                                        # The TTS engine voice that will be mapped to Open AI Echo voice
        self.openai_fable = tts_model_loaded["openai_voices"]["fable"]                                      # The TTS engine voice that will be mapped to Open AI Fable voice
        self.openai_nova = tts_model_loaded["openai_voices"]["nova"]                                        # The TTS engine voice that will be mapped to Open AI Nova voice
        self.openai_onyx = tts_model_loaded["openai_voices"]["onyx"]                                        # The TTS engine voice that will be mapped to Open AI Onyx voice
        self.openai_shimmer = tts_model_loaded["openai_voices"]["shimmer"]                                  # The TTS engine voice that will be mapped to Open AI Shimmer voice
        ###################################################################
        # DONT CHANGE #  Load params and api_defaults from confignew.json #
        ###################################################################
        # Define the path to the confignew.json file
        configfile_path = self.main_dir / "confignew.json"
        # Load config file and get settings
        with open(configfile_path, "r") as configfile:
            configfile_data = json.load(configfile)
        self.branding = configfile_data.get("branding", "")                                                 # Sets up self.branding for outputting the name stored in the "confgnew.json" file, as used in print statements.
        self.params = configfile_data                                                                       # Loads in the curent "confgnew.json" file to self.params.
        self.debug_tts = configfile_data.get("debugging").get("debug_tts")                                  # Can be used within this script as a True/False flag for generally debugging the TTS generation process.
        self.debug_tts_variables = configfile_data.get("debugging").get("debug_tts_variables")              # Can be used within this script as a True/False flag for generally debugging variables (if you wish).

        ############################################################################
        # Fish Speech specific parameters
        ############################################################################
        self.codec_model = None           # DAC codec model for audio encoding/decoding
        self.tokenizer = None             # Fish Speech tokenizer
        self.target_sample_rate = 44100   # Fish Speech uses 44.1kHz
        self.max_new_tokens = 2048        # Maximum tokens to generate
        self.top_p = 0.8                  # Nucleus sampling parameter

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
        # Show device being used
        device_name = self.device.upper()
        if self.device == "mps":
            device_name = "MPS (Metal)"
        print(f"[{self.branding}ENG] \033[92mDevice            :\033[93m {device_name}\033[0m")
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

        # Check if Fish Speech is installed
        global FISH_SPEECH_AVAILABLE
        if not FISH_SPEECH_AVAILABLE:
            print(f"[{self.branding}ENG] \033[93mFish Speech not installed. Installing now...\033[0m")
            install_fish_speech()
            # After installation, try importing again
            try:
                from fish_speech.models.text2semantic.inference import (
                    DualARTransformer,
                    generate_long,
                )
                from fish_speech.models.dac.inference import load_model as load_codec_model
                from fish_speech.text import clean_text
                from fish_speech.tokenizer import FishTokenizer
                FISH_SPEECH_AVAILABLE = True
            except ImportError:
                print(f"[{self.branding}ENG] \033[91mFailed to import Fish Speech after installation\033[0m")
                self.setup_has_run = True
                return

        # Record the start time of loading the model
        generate_start_time = time.time()

        # Scan for available models
        self.available_models = self.scan_models_folder()
        print(f"[{self.branding}ENG] Found models: {list(self.available_models.keys())}") if self.debug_tts else None

        # Load the model if one is selected
        if self.selected_model and self.selected_model != "No Models Found":
            try:
                await self.api_manual_load_model(self.selected_model)
                self.current_model_loaded = self.selected_model
            except Exception as e:
                print(f"[{self.branding}ENG] \033[91mError loading model: {str(e)}\033[0m")
                self.is_tts_model_loaded = False

        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time

        print(f"[{self.branding}ENG]\033[94m Model/Engine :\033[93m Fish Speech\033[94m Ready\033[0m")
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m")

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.setup_has_run = True # Flag that setup has run, so the /api/ready endpoint will send a "Ready" status and load the webui

    ##################################
    ##################################
    # CHANGE ME #  Low VRAM Swapping #
    ##################################
    ##################################
    async def handle_lowvram_change(self):
        """Handle moving model between CPU and GPU for low VRAM operation"""
        if not self.lowvram_capable or not self.cuda_is_available:
            return

        if not self.is_tts_model_loaded:
            return

        try:
            if self.lowvram_enabled and self.device == 'cuda':
                # Moving to CPU
                print(f"[{self.branding}Debug] Moving models to CPU") if self.debug_tts else None

                if self.model is not None:
                    self.model = self.model.to('cpu')
                    torch.cuda.empty_cache()
                    print(f"[{self.branding}Debug] Text model moved to CPU") if self.debug_tts else None

                if self.codec_model is not None:
                    self.codec_model = self.codec_model.to('cpu')
                    torch.cuda.empty_cache()
                    print(f"[{self.branding}Debug] Codec model moved to CPU") if self.debug_tts else None

                self.device = 'cpu'
                print(f"[{self.branding}ENG] Models moved to CPU for low VRAM mode") if self.debug_tts else None

            elif self.lowvram_enabled and self.device == 'cpu':
                # Moving to GPU
                print(f"[{self.branding}Debug] Moving models to GPU") if self.debug_tts else None

                if self.model is not None:
                    self.model = self.model.to('cuda')
                    print(f"[{self.branding}Debug] Text model moved to GPU") if self.debug_tts else None

                if self.codec_model is not None:
                    self.codec_model = self.codec_model.to('cuda')
                    print(f"[{self.branding}Debug] Codec model moved to GPU") if self.debug_tts else None

                self.device = 'cuda'
                print(f"[{self.branding}ENG] Models moved to GPU") if self.debug_tts else None

        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mWarning during model movement: {str(e)}\033[0m")
            if self.lowvram_enabled:
                self.device = 'cpu'
            else:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ########################################
    ########################################
    # CHANGE ME #  DeepSpeed model loading #
    ########################################
    ########################################
    async def handle_deepspeed_change(self, value):
        """Fish Speech currently doesn't support DeepSpeed"""
        print(f"[{self.branding}ENG] DeepSpeed not supported for Fish Speech")
        return False


    #####################################################################################
    #####################################################################################
    # CHANGE ME # scan for available models/voices that are relevant to this TTS engine #
    #####################################################################################
    #####################################################################################
    def scan_models_folder(self):
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        """Scan for available Fish Speech models"""
        self.available_models = {}
        models_dir = self.main_dir / "models" / "fishspeech"

        if not models_dir.exists():
            print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Models directory not found: {models_dir}")
            print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Please use the Gradio interface to download a model.")
            self.available_models["No Models Found"] = "fishspeech"
            return self.available_models

        # Look for model directories
        found_valid_model = False
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Check for required files
                model_file = model_dir / "model.pth"
                config_file = model_dir / "config.json"
                tokenizer_file = model_dir / "tokenizer.tiktoken"

                # Check for vocoder/codec - supports both formats
                codec_file = model_dir / "codec.pth"
                firefly_files = list(model_dir.glob("firefly-gan-vq-*.pth"))
                has_vocoder = codec_file.exists() or len(firefly_files) > 0

                if model_file.exists() and config_file.exists() and tokenizer_file.exists() and has_vocoder:
                    model_name = model_dir.name
                    self.available_models[f"fishspeech - {model_name}"] = "fishspeech"
                    found_valid_model = True
                    vocoder_type = "codec.pth" if codec_file.exists() else firefly_files[0].name
                    print(f"[{self.branding}ENG] Found valid model: {model_name} (vocoder: {vocoder_type})") if self.debug_tts else None
                else:
                    missing = []
                    if not model_file.exists():
                        missing.append("model.pth")
                    if not has_vocoder:
                        missing.append("codec.pth or firefly-gan-vq-*.pth")
                    if not config_file.exists():
                        missing.append("config.json")
                    if not tokenizer_file.exists():
                        missing.append("tokenizer.tiktoken")
                    if missing:
                        print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Model folder '{model_dir.name}' is missing: {', '.join(missing)}")

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        if not found_valid_model:
            self.available_models["No Models Found"] = "fishspeech"
            print(f"[{self.branding}ENG] \033[91mWarning\033[0m: No valid Fish Speech models found")
        return self.available_models

    #############################################################
    #############################################################
    # CHANGE ME #  POPULATE FILES LIST FROM VOICES DIRECTORY ####
    #############################################################
    #############################################################
    def voices_file_list(self):
        """List available voices - wav files with matching .reference.txt files"""
        try:
            voices = []

            def has_reference_text(wav_path):
                """Check if a wav file has a corresponding reference text file"""
                text_path = wav_path.with_suffix('.reference.txt')
                return text_path.exists()

            directory = self.main_dir / "voices"

            # Add .wav files in the main "voices" directory (only if they have matching .reference.txt)
            for f in directory.glob("*.wav"):
                if has_reference_text(f):
                    voices.append(f.name)
                else:
                    print(f"[{self.branding}ENG] Warning: {f.name} does not have a matching reference text file") if self.debug_tts else None

            # Walk through subfolders and add subfolder names if they contain valid wav+reference.txt pairs
            for folder in directory.iterdir():
                if folder.is_dir():
                    valid_pairs = False
                    for wav_file in folder.glob("*.wav"):
                        if has_reference_text(wav_file):
                            valid_pairs = True
                            break

                    if valid_pairs:
                        folder_name = folder.name + "/"
                        voices.append(folder_name)
                    elif self.debug_tts:
                        print(f"[{self.branding}ENG] Warning: Folder {folder.name} has no valid wav+reference.txt pairs")

            # Remove "voices/" from the list if it somehow got added
            voices = [v for v in voices if v != "voices/"]

            if not voices:
                return ["No Voices Found"]
            return voices
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Voices/Voice Models not found. Cannot load a list of voices.")
            print(f"[{self.branding}ENG]")
            return ["No Voices Found"]

    #################################################################################
    #################################################################################
    # CHANGE ME # Model loading #####################################################
    #################################################################################
    #################################################################################
    async def api_manual_load_model(self, model_name):
        if model_name == "No Models Found":
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load.")
            raise HTTPException(status_code=400, detail="No models for this TTS engine were found to load.")

        print(f"[{self.branding}ENG]\033[94m Model/Engine :\033[93m {model_name}\033[94m loading into\033[93m", self.device,"\033[0m")

        # Split the engine name from model name and get just the model folder name
        model_folder = model_name.split(" - ")[-1]
        model_dir = self.main_dir / "models" / "fishspeech" / model_folder

        if not model_dir.exists():
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Model directory not found: {model_dir}")
            print(f"[{self.branding}ENG] \033[93mPlease download a Fish Speech model in the Gradio interface.\033[0m")
            raise HTTPException(status_code=404, detail=f"Model directory not found: {model_dir}")

        try:
            # Import Fish Speech components
            from fish_speech.models.text2semantic.llama import DualARTransformer

            # Find vocoder - supports both codec.pth and firefly-gan-vq-*.pth
            codec_path = model_dir / "codec.pth"
            firefly_files = list(model_dir.glob("firefly-gan-vq-*.pth"))

            if codec_path.exists():
                vocoder_path = codec_path
                vocoder_type = "codec"
            elif firefly_files:
                vocoder_path = firefly_files[0]
                vocoder_type = "firefly"
            else:
                raise FileNotFoundError("No vocoder found (codec.pth or firefly-gan-vq-*.pth)")

            print(f"[{self.branding}ENG] Loading model from: {model_dir}") if self.debug_tts else None
            print(f"[{self.branding}ENG] Using vocoder: {vocoder_path.name}") if self.debug_tts else None

            # Store vocoder type for later use
            self.vocoder_type = vocoder_type

            # Warn about experimental fish-speech-1.5 support
            if vocoder_type == "firefly":
                print(f"[{self.branding}ENG] \033[93mNote: fish-speech-1.5 support is experimental. For best results, use openaudio-s1-mini.\033[0m")

            # Determine precision - MPS requires FP32
            if self.device == "mps":
                precision = torch.float32
            elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                precision = torch.bfloat16
            else:
                precision = torch.float16

            # Use official from_pretrained method which handles everything correctly
            print(f"[{self.branding}ENG] Loading text-to-semantic model...")
            self.model = DualARTransformer.from_pretrained(
                str(model_dir),
                load_weights=True
            )
            self.tokenizer = self.model.tokenizer

            # Apply precision and move to device
            self.model = self.model.to(dtype=precision, device=self.device)
            self.model.eval()
            self.precision = precision  # Store for later use

            # Initialize KV caches for generation - CRITICAL for proper output
            print(f"[{self.branding}ENG] Initializing model caches...")
            with torch.device(self.device):
                self.model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=self.model.config.max_seq_len,
                    dtype=precision,
                )

            # Load vocoder/codec model based on type
            print(f"[{self.branding}ENG] Loading vocoder ({vocoder_type})...")
            if vocoder_type == "codec":
                from fish_speech.models.dac.inference import load_model as load_codec_model
                self.codec_model = load_codec_model(
                    config_name="modded_dac_vq",
                    checkpoint_path=str(vocoder_path),
                    device=self.device
                )
            else:
                # Firefly GAN vocoder for fish-speech-1.5
                from fish_speech.models.vqgan.inference import load_model as load_firefly_model
                self.codec_model = load_firefly_model(
                    config_name="firefly_gan_vq",
                    checkpoint_path=str(vocoder_path),
                    device=self.device
                )

            # Ensure codec model uses correct precision for MPS
            if self.device == "mps":
                self.codec_model = self.codec_model.to(dtype=torch.float32)

            self.is_tts_model_loaded = True
            print(f"[{self.branding}ENG] Fish Speech model loaded successfully")

        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError loading model: {str(e)}\033[0m")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    ###############################
    ###############################
    # CHANGE ME # Model unloading #
    ###############################
    ###############################
    async def unload_model(self):
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        """Unload the Fish Speech model from memory"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, 'codec_model') and self.codec_model is not None:
            del self.codec_model
            self.codec_model = None

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.is_tts_model_loaded = False
        return None

    ###################################################################################################################################
    ###################################################################################################################################
    # CHANGE ME # Model changing. Unload out old model and load in a new one ##########################################################
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

        # Unload current model if one is loaded
        if self.is_tts_model_loaded:
            await self.unload_model()

        # Load the new model
        try:
            await self.api_manual_load_model(tts_method)
            self.current_model_loaded = tts_method
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError loading model: {str(e)}\033[0m")
            return False

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m")
        return True


    async def preprocess_ref_audio_text(self, ref_audio_orig, ref_text, max_duration=30):
        """Preprocess reference audio - clip to max duration and clean up"""
        print(f"[{self.branding}ENG] Preprocessing reference audio...") if self.debug_tts else None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            aseg = AudioSegment.from_file(ref_audio_orig)

            # Clip to max duration (Fish Speech recommends 10-30 seconds)
            if len(aseg) > max_duration * 1000:
                # Try to find a good cut point at silence
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=500, silence_thresh=-40, keep_silence=500, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave + non_silent_seg) > max_duration * 1000:
                        break
                    non_silent_wave += non_silent_seg

                if len(non_silent_wave) > 0:
                    aseg = non_silent_wave
                else:
                    aseg = aseg[:max_duration * 1000]

                print(f"[{self.branding}ENG] Clipped audio to {len(aseg)/1000:.1f}s") if self.debug_tts else None

            # Remove silence from edges
            aseg = self.remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
            aseg.export(f.name, format="wav")
            ref_audio = f.name

        # Ensure ref_text ends with proper punctuation
        if not ref_text.endswith(". ") and not ref_text.endswith("。"):
            if ref_text.endswith("."):
                ref_text += " "
            else:
                ref_text += ". "

        return ref_audio, ref_text

    def convert_emotion_markers(self, text):
        """
        Convert :marker: syntax to (marker) syntax for Fish Speech emotion control.

        AllTalk's text filter strips parentheses, so users can use :marker: format
        which survives filtering. This function converts them back to (marker) format
        that Fish Speech expects.

        Supported markers (from OpenAudio S1 documentation):
        - Emotions: :angry:, :sad:, :excited:, :surprised:, etc.
        - Tones: :whispering:, :shouting:, :soft tone:, etc.
        - Effects: :laughing:, :sighing:, :sobbing:, etc.

        Usage: "Hello :whispering: this is a secret :excited: isn't this great?"
        Becomes: "Hello (whispering) this is a secret (excited) isn't this great?"
        """
        import re

        # Sanitize inappropriate markers
        text = re.sub(r':moan:', ':scream:', text, flags=re.IGNORECASE)
        text = re.sub(r'\(moan\)', '(scream)', text, flags=re.IGNORECASE)
        text = re.sub(r':gasp:', ':sigh:', text, flags=re.IGNORECASE)
        text = re.sub(r'\(gasp\)', '(sigh)', text, flags=re.IGNORECASE)

        # Pattern matches :word: or :word word: (for multi-word markers like "soft tone")
        # The marker must be alphanumeric with optional spaces between words
        pattern = r':([a-zA-Z][a-zA-Z0-9 ]*[a-zA-Z0-9]):'

        def replace_marker(match):
            marker = match.group(1)
            return f'({marker})'

        converted = re.sub(pattern, replace_marker, text)

        if converted != text:
            print(f"[{self.branding}ENG] Converted emotion markers: {text[:50]}... -> {converted[:50]}...")

        return converted

    def remove_silence_edges(self, audio, silence_threshold=-42):
        """Remove silence from the start and end of audio"""
        try:
            # Remove silence from the start
            non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
            audio = audio[non_silent_start_idx:]

            # Remove silence from the end
            non_silent_end_duration = audio.duration_seconds
            for ms in reversed(audio):
                if ms.dBFS > silence_threshold:
                    break
                non_silent_end_duration -= 0.001
            trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

            return trimmed_audio
        except Exception as e:
            print(f"[{self.branding}ENG] Error in remove_silence_edges: {str(e)}") if self.debug_tts else None
            return audio


    async def encode_reference_audio(self, ref_audio_path):
        """Encode reference audio to tokens using the codec model"""
        # Load audio
        audio, sr = torchaudio.load(ref_audio_path)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)

        # Add batch dimension: [1, 1, samples] and use correct precision
        audio = audio.unsqueeze(0).to(device=self.device, dtype=self.precision if hasattr(self, 'precision') else torch.float32)

        # Encode to tokens
        audio_lengths = torch.tensor([audio.shape[2]], device=self.device, dtype=torch.long)
        with torch.no_grad():
            indices, indices_lens = self.codec_model.encode(audio, audio_lengths)

        # Return as tensor with shape [num_codebooks, seq_len]
        # generate_long expects codes where codes[0] is a 1D tensor for the first codebook
        # indices shape is typically [batch, num_codebooks, seq_len], so squeeze batch dim
        if indices.dim() == 3:
            indices = indices.squeeze(0)  # Remove batch dimension
        return indices.cpu()


    async def decode_tokens_to_audio(self, tokens):
        """Decode generated tokens back to audio using the codec model"""
        # Handle both numpy arrays and torch tensors
        if isinstance(tokens, np.ndarray):
            indices = torch.from_numpy(tokens).to(self.device).long()
        else:
            indices = tokens.to(self.device).long()

        print(f"[{self.branding}Debug] Decode input shape: {indices.shape}")
        print(f"[{self.branding}Debug] Decode input dtype: {indices.dtype}")

        with torch.no_grad():
            if self.vocoder_type == "firefly":
                # Firefly vocoder expects 3D: [batch, codebooks, seq_len]
                if indices.dim() == 2:
                    indices = indices.unsqueeze(0)  # Add batch dimension

                seq_len = indices.shape[2]
                feature_lengths = torch.tensor([seq_len], device=self.device, dtype=torch.long)
                print(f"[{self.branding}Debug] Firefly decode - indices shape: {indices.shape}, feature_lengths: {feature_lengths}")

                audio, audio_lengths = self.codec_model.decode(indices=indices, feature_lengths=feature_lengths)
            else:
                # DAC codec expects 2D: [num_codebooks, seq_len]
                if indices.dim() == 3:
                    indices = indices.squeeze(0)

                assert indices.dim() == 2, f"Expected 2D indices for codec, got {indices.dim()}"

                seq_len = indices.shape[1]
                indices_lens = torch.tensor([seq_len], device=self.device, dtype=torch.long)
                print(f"[{self.branding}Debug] Codec decode - indices shape: {indices.shape}, indices_lens: {indices_lens}")

                audio, audio_lengths = self.codec_model.decode(indices, indices_lens)

        print(f"[{self.branding}Debug] Decoded audio shape: {audio.shape}")

        # Extract mono audio - output is [batch, channels, samples]
        audio = audio[0, 0].float().cpu().numpy()

        # Get sample rate from the vocoder model
        if self.vocoder_type == "firefly":
            sample_rate = self.codec_model.spec_transform.sample_rate
        else:
            sample_rate = self.codec_model.sample_rate

        return audio, sample_rate


    ##########################################################################################################################################
    ##########################################################################################################################################
    # CHANGE ME # TTS Generation #############################################################################################################
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

        # Debug: Show model configuration
        print(f"[{self.branding}Debug] Model loaded: {self.current_model_loaded}")
        print(f"[{self.branding}Debug] Vocoder type: {self.vocoder_type}")
        print(f"[{self.branding}Debug] Model num_codebooks: {self.model.config.num_codebooks}")
        print(f"[{self.branding}Debug] Model codebook_size: {self.model.config.codebook_size}")

        try:
            # Move to GPU if using low VRAM mode
            if self.lowvram_enabled and self.device == "cpu":
                print(f"[{self.branding}Debug] Moving models to GPU for generation") if self.debug_tts else None
                await self.handle_lowvram_change()

            # Handle voice folder or direct file
            if voice.endswith('/'):
                # It's a folder, pick the first valid wav+reference.txt pair
                voice_dir = self.main_dir / "voices" / voice.rstrip('/')
                for wav_file in voice_dir.glob("*.wav"):
                    ref_text_path = wav_file.with_suffix('.reference.txt')
                    if ref_text_path.exists():
                        ref_audio_path = wav_file
                        break
            else:
                # It's a direct file
                ref_audio_path = self.main_dir / "voices" / voice

            # Get the corresponding reference text file
            ref_text_path = ref_audio_path.with_suffix('.reference.txt')
            if not ref_text_path.exists():
                raise FileNotFoundError(f"Reference text file not found for {voice}")

            with open(ref_text_path, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()

            # Preprocess reference audio and text
            ref_audio, processed_ref_text = await self.preprocess_ref_audio_text(
                str(ref_audio_path), ref_text
            )

            print(f"[{self.branding}Debug] Reference text: {processed_ref_text[:50]}...") if self.debug_tts else None
            print(f"[{self.branding}Debug] Target text: {text[:50]}...") if self.debug_tts else None

            # Encode reference audio to tokens
            print(f"[{self.branding}ENG] Encoding reference audio...") if self.debug_tts else None
            prompt_tokens = await self.encode_reference_audio(ref_audio)

            # Verify prompt_tokens matches model's num_codebooks
            expected_codebooks = self.model.config.num_codebooks
            actual_codebooks = prompt_tokens.shape[0]
            print(f"[{self.branding}Debug] Prompt tokens shape: {prompt_tokens.shape}")
            print(f"[{self.branding}Debug] Expected codebooks: {expected_codebooks}, Actual: {actual_codebooks}")
            if actual_codebooks != expected_codebooks:
                print(f"[{self.branding}ENG] \033[91mWarning: Codebook mismatch! Model expects {expected_codebooks} but codec produced {actual_codebooks}\033[0m")

            # Import generation function
            from fish_speech.models.text2semantic.inference import generate_long, decode_one_token_ar

            # Generate semantic tokens
            print(f"[{self.branding}ENG] Generating speech...")

            # Use Fish Speech's generation
            temp = float(temperature) if temperature else self.temperature_set
            rep_penalty = float(repetition_penalty) if repetition_penalty else self.repetitionpenalty_set

            # Reset KV caches before generation to clear stale state
            # Must be done in inference_mode since caches were created there
            with torch.inference_mode():
                for layer in self.model.layers:
                    if hasattr(layer.attention, 'kv_cache') and layer.attention.kv_cache is not None:
                        layer.attention.kv_cache.k_cache.zero_()
                        layer.attention.kv_cache.v_cache.zero_()
                # Also reset fast layers if they exist
                if hasattr(self.model, 'fast_layers'):
                    for layer in self.model.fast_layers:
                        if hasattr(layer.attention, 'kv_cache') and layer.attention.kv_cache is not None:
                            layer.attention.kv_cache.k_cache.zero_()
                            layer.attention.kv_cache.v_cache.zero_()

            # Convert :emotion: markers to (emotion) format for Fish Speech
            # AllTalk's text filter strips parentheses, so users can use :marker: format
            text = self.convert_emotion_markers(text)

            # Calculate appropriate max_new_tokens based on text length
            # Roughly 20 tokens per second of audio, 2-3 words per second
            # So ~7-10 tokens per word. Use 15 tokens/word with 3x buffer
            word_count = len(text.split())
            estimated_tokens = word_count * 15 * 3  # 3x buffer for safety
            max_tokens = min(max(estimated_tokens, 256), self.max_new_tokens)  # At least 256, at most max_new_tokens
            print(f"[{self.branding}Debug] Text word count: {word_count}, Max tokens: {max_tokens}")

            generated_tokens = None
            for response in generate_long(
                model=self.model,
                decode_one_token=decode_one_token_ar,
                device=self.device,
                text=text,
                prompt_text=processed_ref_text,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=self.top_p,
                repetition_penalty=rep_penalty,
            ):
                if response.codes is not None:
                    generated_tokens = response.codes
                    print(f"[{self.branding}Debug] Generated codes shape: {generated_tokens.shape}")
                    print(f"[{self.branding}Debug] Generated codes dtype: {generated_tokens.dtype}")
                    print(f"[{self.branding}Debug] Generated codes min/max: {generated_tokens.min().item()}/{generated_tokens.max().item()}")

            if generated_tokens is None:
                raise RuntimeError("No tokens were generated")

            # Decode tokens to audio
            print(f"[{self.branding}ENG] Decoding audio...")
            print(f"[{self.branding}Debug] Prompt tokens shape: {prompt_tokens.shape}")
            print(f"[{self.branding}Debug] Prompt tokens min/max: {prompt_tokens.min().item()}/{prompt_tokens.max().item()}")
            final_wave, sample_rate = await self.decode_tokens_to_audio(generated_tokens)

            # Apply speed adjustment if needed
            if speed and float(speed) != 1.0:
                # Use torchaudio or pydub for speed adjustment
                speed_factor = float(speed)
                # Simple resampling for speed change
                if speed_factor != 1.0:
                    import scipy.signal as signal
                    final_wave = signal.resample(
                        final_wave,
                        int(len(final_wave) / speed_factor)
                    )

            # Save the audio
            sf.write(output_file, final_wave, sample_rate)

            # Clean up temp file
            try:
                os.unlink(ref_audio)
            except OSError:
                pass

            generate_end_time = time.time()
            generate_elapsed_time = generate_end_time - generate_start_time
            print(f"[{self.branding}GEN] \033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{self.lowvram_enabled} \033[94mDeepSpeed: \033[33m{self.deepspeed_enabled}\033[0m")

            if streaming:
                with open(output_file, 'rb') as f:
                    yield f.read()

        except FileNotFoundError as e:
            print(f"[{self.branding}ENG] \033[91mError: {str(e)}\033[0m")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError during TTS generation: {str(e)}\033[0m")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error during TTS generation: {str(e)}")
        finally:
            try:
                # Handle low VRAM mode - move back to CPU if needed
                if self.lowvram_enabled and not self.tts_narrator_generatingtts:
                    print(f"[{self.branding}Debug] Moving models back to CPU after generation") if self.debug_tts else None
                    await self.handle_lowvram_change()
            except Exception as e:
                print(f"[{self.branding}Debug] Device movement warning: {str(e)}") if self.debug_tts else None

            self.tts_generating_lock = False
