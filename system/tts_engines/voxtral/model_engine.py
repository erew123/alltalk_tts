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
import base64
import subprocess
import requests
import io

# Voxtral TTS is an HTTP-client engine — no model loaded in memory.
# It sends requests to either a local vllm-omni server or the Mistral Cloud API,
# both exposing an OpenAI-compatible /v1/audio/speech endpoint.

MLX_MODEL_VARIANTS = {
    "bf16": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
    "6bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit",
    "4bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
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
        self.this_dir = Path(__file__).parent.resolve()                         # Sets up self.this_dir as a variable for the folder THIS script is running in.
        self.main_dir = Path(__file__).parent.parent.parent.parent.resolve()    # Sets up self.main_dir as a variable for the folder AllTalk is running in
        self.device = "cuda" if torch.cuda.is_available() else "cpu"            # Sets up self.device to cuda if torch exists with Nvidia/CUDA, otherwise sets to cpu
        self.cuda_is_available = torch.cuda.is_available()                      # Sets up cuda_is_available as a True/False to track if Nvidia/CUDA was found on the system
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
        #####################################
        # Voxtral-specific settings         #
        #####################################
        self.backend_type = tts_model_loaded["settings"].get("backend_type", "mlx-audio")  # "mlx-audio", "vllm-omni", or "mistral-cloud"
        self.api_url = tts_model_loaded["settings"].get("api_url", "http://localhost:7853")
        self.api_key = tts_model_loaded["settings"].get("api_key", "")
        self.response_format = tts_model_loaded["settings"].get("response_format", "wav")
        self.auto_start_server = tts_model_loaded["settings"].get("auto_start_server", True)
        self.server_port = tts_model_loaded["settings"].get("server_port", 7853)
        self.mlx_model_variant = tts_model_loaded["settings"].get("mlx_model_variant", "bf16")
        self._server_process = None

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

        generate_start_time = time.time()
        self.available_models = self.scan_models_folder()
        print(f"[{self.branding}ENG] \033[92mVoxtral Backend   :\033[93m {self.backend_type}\033[0m")
        if self.backend_type == "mlx-audio":
            print(f"[{self.branding}ENG] \033[92mMLX Model Variant :\033[93m {self.mlx_model_variant} ({MLX_MODEL_VARIANTS.get(self.mlx_model_variant, 'unknown')})\033[0m")
        print(f"[{self.branding}ENG] \033[92mVoxtral API URL   :\033[93m {self.api_url}\033[0m")
        print(f"[{self.branding}ENG] \033[92mVoxtral API Key   :\033[93m {'Set' if self.api_key else 'Not set (local mode)'}\033[0m")
        # Auto-start mlx-audio server if configured
        self._start_mlx_server()
        # Validate API connectivity
        api_ok = self._test_api_connection()
        if api_ok:
            print(f"[{self.branding}ENG] \033[92mVoxtral API       :\033[93m Connected\033[0m")
        else:
            print(f"[{self.branding}ENG] \033[91mVoxtral API       : Not reachable (will retry on generation)\033[0m")
        self.current_model_loaded = "voxtral - voxtral-tts-2603"
        self.is_tts_model_loaded = True
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[{self.branding}ENG]\033[94m Model/Engine :\033[93m Voxtral TTS\033[94m Ready\033[0m")
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m")

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.setup_has_run = True # Flag that setup has run, so the /api/ready endpoint will send a "Ready" status and load the webui

    def _test_api_connection(self):
        """Ping the Voxtral API to check connectivity."""
        try:
            url = self.api_url.rstrip("/")
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            # Try the models endpoint first (vllm-omni), fall back to a simple GET
            for endpoint in [f"{url}/v1/models", f"{url}/health", url]:
                try:
                    resp = requests.get(endpoint, headers=headers, timeout=5)
                    if resp.status_code < 500:
                        return True
                except requests.exceptions.RequestException:
                    continue
            return False
        except Exception:
            return False

    def _start_mlx_server(self):
        """Auto-start the mlx-audio server if backend is mlx-audio and auto_start_server is enabled."""
        if self.backend_type != "mlx-audio" or not self.auto_start_server:
            return
        port = self.server_port
        # Check if a server is already responding on the port
        try:
            resp = requests.get(f"http://localhost:{port}/v1/models", timeout=3)
            if resp.status_code < 500:
                print(f"[{self.branding}ENG] \033[92mmlx-audio server already running on port {port}, reusing.\033[0m")
                self.api_url = f"http://localhost:{port}"
                return
        except requests.exceptions.RequestException:
            pass
        # Launch the mlx-audio server
        model_id = MLX_MODEL_VARIANTS.get(self.mlx_model_variant, MLX_MODEL_VARIANTS["bf16"])
        print(f"[{self.branding}ENG] \033[93mStarting mlx-audio server on port {port} with model {model_id}...\033[0m")
        log_path = os.path.join(self.this_dir, "mlx_server.log")
        log_file = open(log_path, "w")
        try:
            self._server_process = subprocess.Popen(
                [
                    sys.executable, "-m", "mlx_audio.server",
                   # "--model", model_id,
                    "--port", str(port),
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except Exception as e:
            log_file.close()
            print(f"[{self.branding}ENG] \033[91mFailed to start mlx-audio server: {e}\033[0m")
            return
        # Poll health endpoint until the server is ready (up to 120s for first-run model download)
        print(f"[{self.branding}ENG] \033[93mWaiting for mlx-audio server to be ready (first run may download model)...\033[0m")
        timeout = 120
        interval = 1
        elapsed = 0
        while elapsed < timeout:
            # Check if the process died
            if self._server_process.poll() is not None:
                print(f"[{self.branding}ENG] \033[91mmlx-audio server exited unexpectedly (code {self._server_process.returncode}). Check {log_path}\033[0m")
                self._server_process = None
                return
            try:
                resp = requests.get(f"http://localhost:{port}/v1/models", timeout=3)
                if resp.status_code < 500:
                    self.api_url = f"http://localhost:{port}"
                    print(f"[{self.branding}ENG] \033[92mmlx-audio server ready on port {port} (took {elapsed}s)\033[0m")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(interval)
            elapsed += interval
        print(f"[{self.branding}ENG] \033[91mmlx-audio server did not become ready within {timeout}s. Check {log_path}\033[0m")

    def _stop_mlx_server(self):
        """Stop the auto-started mlx-audio server subprocess, if any."""
        if self._server_process is None:
            return
        if self._server_process.poll() is not None:
            # Already exited
            self._server_process = None
            return
        print(f"[{self.branding}ENG] \033[93mStopping mlx-audio server (pid {self._server_process.pid})...\033[0m")
        try:
            self._server_process.terminate()
            self._server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"[{self.branding}ENG] \033[91mmlx-audio server did not stop gracefully, killing...\033[0m")
            self._server_process.kill()
            self._server_process.wait(timeout=5)
        self._server_process = None
        print(f"[{self.branding}ENG] \033[92mmlx-audio server stopped.\033[0m")

    ##################################
    ##################################
    # CHANGE ME #  Low VRAM Swapping #
    ##################################
    ##################################
    async def handle_lowvram_change(self):
        pass # Voxtral is an HTTP-client engine, no model in VRAM

    ########################################
    ########################################
    # CHANGE ME #  DeepSpeed model loading #
    ########################################
    ########################################
    async def handle_deepspeed_change(self, value):
        pass # Voxtral is an HTTP-client engine, DeepSpeed not applicable

    #####################################################################################
    #####################################################################################
    # CHANGE ME # scan for available models/voices that are relevant to this TTS engine #
    #####################################################################################
    #####################################################################################
    def scan_models_folder(self):
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        self.available_models = {'voxtral - voxtral-tts-2603': 'voxtral-tts-2603'}

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        if not self.available_models:
            self.available_models["No Models Found"] = "No Models Found"
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

            voices_file = os.path.join(self.this_dir, "voxtral_voices.json")
            if os.path.exists(voices_file):
                with open(voices_file, "r") as f:
                    voices_data = json.load(f)
                    voices = [voice["voice_code"] for voice in voices_data["voices"]]

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # ↓↓↓ Keep everything below this line ↓↓↓
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            if not voices:
                return ["No Voices Found"]
            return sorted(voices)
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Voices/Voice Models not found. Cannot load a list of voices.")
            print(f"[{self.branding}ENG]")
            return ["No Voices Found"]

    #################################################################################
    #################################################################################
    # CHANGE ME # Model loading # Voxtral does not load a model into memory         #
    #################################################################################
    #################################################################################
    async def api_manual_load_model(self, model_name):
        if model_name == "No Models Found":
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load.")
            raise HTTPException(status_code=400, detail="No models for this TTS engine were found to load.")
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


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

        self._stop_mlx_server()

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.is_tts_model_loaded = False
        return None

    ###################################################################################################################################
    ###################################################################################################################################
    # CHANGE ME # Model changing. Unload out old model and load in a new one                                                          #
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


        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m")
        return True

    ##########################################################################################################################################
    ##########################################################################################################################################
    # CHANGE ME # TTS Generation — POST to /v1/audio/speech endpoint                                                                        #
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
            # Reload settings in case they were changed via the settings page
            with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
                current_settings = json.load(f)
            backend_type = current_settings["settings"].get("backend_type", self.backend_type)
            api_url = current_settings["settings"].get("api_url", self.api_url).rstrip("/")
            api_key = current_settings["settings"].get("api_key", self.api_key)
            response_format = current_settings["settings"].get("response_format", self.response_format)
            mlx_variant = current_settings["settings"].get("mlx_model_variant", self.mlx_model_variant)

            # Build request headers
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            # Build request payload based on backend type
            # All three backends expose /v1/audio/speech but differ in model name and voice parameter
            endpoint = f"{api_url}/v1/audio/speech"

            if backend_type == "mistral-cloud":
                payload = {
                    "model": "voxtral-tts-2603",
                    "input": text,
                    "voice_id": voice,
                    "response_format": response_format,
                }
            elif backend_type == "mlx-audio":
                payload = {
                    "model": MLX_MODEL_VARIANTS.get(mlx_variant, MLX_MODEL_VARIANTS["bf16"]),
                    "input": text,
                    "voice": voice,
                    "response_format": response_format,
                }
            else:
                # vllm-omni
                payload = {
                    "model": "mistralai/Voxtral-4B-TTS-2603",
                    "input": text,
                    "voice": voice,
                    "response_format": response_format,
                }

            # Add speed if supported and not default
            if speed and float(speed) != 1.0:
                payload["speed"] = float(speed)

            print(f"[{self.branding}Debug] Voxtral request: voice={voice}, format={response_format}, endpoint={endpoint}") if self.debug_tts else None

            # Check if voice is a path to a reference audio file (voice cloning)
            ref_audio_path = None
            if voice and os.path.isfile(voice):
                ref_audio_path = voice

            if ref_audio_path:
                # Voice cloning: encode reference audio as base64
                with open(ref_audio_path, "rb") as af:
                    audio_bytes = af.read()
                ref_audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                if backend_type == "mistral-cloud":
                    payload.pop("voice_id", None)
                    payload["ref_audio"] = ref_audio_b64
                else:
                    # mlx-audio and vllm-omni both use voice_prompt
                    payload.pop("voice", None)
                    payload["voice_prompt"] = ref_audio_b64
                print(f"[{self.branding}Debug] Using reference audio for voice cloning: {ref_audio_path}") if self.debug_tts else None

            if streaming:
                # Streaming mode — yield chunks as they arrive
                resp = requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=120)
                if resp.status_code != 200:
                    error_detail = resp.text[:500]
                    print(f"[{self.branding}ENG] \033[91mError\033[0m: Voxtral API returned {resp.status_code}: {error_detail}")
                    raise HTTPException(status_code=resp.status_code, detail=f"Voxtral API error: {error_detail}")
                # Write streaming response to file and yield chunks
                with open(output_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=4096):
                        if self.tts_stop_generation:
                            self.tts_stop_generation = False
                            break
                        if chunk:
                            f.write(chunk)
                            yield chunk
            else:
                # Non-streaming mode — get full response
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
                if resp.status_code != 200:
                    error_detail = resp.text[:500]
                    print(f"[{self.branding}ENG] \033[91mError\033[0m: Voxtral API returned {resp.status_code}: {error_detail}")
                    raise HTTPException(status_code=resp.status_code, detail=f"Voxtral API error: {error_detail}")
                # Save response audio bytes to output file
                audio_data = resp.content
                with open(output_file, "wb") as f:
                    f.write(audio_data)

        except requests.exceptions.Timeout:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Voxtral API request timed out.")
            raise HTTPException(status_code=504, detail="Voxtral API request timed out.")
        except requests.exceptions.ConnectionError:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Cannot connect to Voxtral API at {api_url}. Is the server running?")
            raise HTTPException(status_code=503, detail=f"Cannot connect to Voxtral API at {api_url}.")
        except HTTPException:
            raise
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Voxtral generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Voxtral generation failed: {str(e)}")

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[{self.branding}GEN] \033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{self.lowvram_enabled} \033[94mDeepSpeed: \033[33m{self.deepspeed_enabled}\033[0m")
        self.tts_generating_lock = False
