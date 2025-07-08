###############################################
# DONT CHANGE # These are base imports needed #
###############################################
import os
import sys
import json
import time
import torch
import random
import numpy as np
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
# In this section you will import any imports that your specific TTS Engine will use. You will provide any
# start-up errors for those bits, as if you were starting up a normal Python script. Note the logging.disable
# a few lines up from here, you may want to # that out while debugging!

import torchaudio as ta
import tempfile
import subprocess

def install_and_restart():
    try:
        print("#################################################################")
        print("Installing required packages for Chatterbox TTS... This may take a while.")
        print("################################################################")
        
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print("Installing ChatterboxTTS with PyTorch CUDA stack...")
            try:
                # Get current torch version to match
                torch_version = torch.__version__.split('+')[0]  # Remove +cu118 suffix if present
                print(f"Installing PyTorch CUDA stack version: {torch_version}")

                # Install ChatterboxTTS and dependencies
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "pip", 
                    "install", 
                    "chatterbox-tts",
                    "transformers",
                    "accelerate",
                    "soundfile",
                    "librosa"
                ])
                
                # Install PyTorch CUDA stack + ChatterboxTTS
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "pip", 
                    "install", 
                    "--upgrade", 
                    "--index-url", 
                    "https://download.pytorch.org/whl/cu118",
                    f"torch=={torch_version}",
                    f"torchaudio=={torch_version}",
                    "torchvision"
                ])
                
                
                print("CUDA installation completed successfully!")
                
            except subprocess.CalledProcessError:
                print("Failed to install CUDA version, falling back to CPU installation...")
                # Fallback to CPU installation
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "pip", 
                    "install", 
                    "chatterbox-tts"
                ])
                print("CPU fallback installation completed!")
        else:
            print("Installing ChatterboxTTS (CPU version)...")
            # Just install ChatterboxTTS for CPU systems
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "chatterbox-tts"
            ])
            print("CPU installation completed successfully!")
        
        print("##############################################################")
        print("All packages installed successfully! Restarting application...")
        print("##############################################################")
        
        # Get the current script's path
        script_path = sys.argv[0]
        
        # Restart the current script
        os.execv(sys.executable, ['python'] + sys.argv)
        
    except subprocess.CalledProcessError as e:
        print("########################################################")
        print(f"Failed to install required packages: {str(e)}")
        print("########################################################")
        print("Trying alternative installation method...")
        
        # Fallback: Try installing with --upgrade-strategy only-if-needed
        try:
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "--upgrade-strategy", 
                "only-if-needed",
                "chatterbox-tts"
            ])
            print("Alternative installation succeeded!")
            os.execv(sys.executable, ['python'] + sys.argv)
        except subprocess.CalledProcessError as e2:
            print("########################################################")
            print(f"Alternative installation also failed: {str(e2)}")
            print("########################################################")
            print("Please manually install chatterbox-tts:")
            print("  pip install --upgrade-strategy only-if-needed chatterbox-tts")
            print("If you have CUDA, also install:")
            print("  pip install --index-url https://download.pytorch.org/whl/cu118 torchaudio")
            print("Or use a virtual environment to avoid conflicts.")
            raise ImportError("Could not install required packages")

try:
    from chatterbox.tts import ChatterboxTTS
except ImportError as IE:
    # raise IE
    install_and_restart()

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
        
        ############################################################################
        # Chatterbox TTS specific parameters
        ############################################################################
        # Automatically detect the best available device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

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
        print(f"[{self.branding}ENG] \033[92mDevice            :\033[93m {self.device}\033[0m")
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

        # Scan available models first
        self.available_models = self.scan_models_folder()
        
        print(f"[{self.branding}ENG] \033[92mLoading Chatterbox TTS model...\033[0m")
        try:
            # Load the ChatterboxTTS model with device detection
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.is_tts_model_loaded = True
            self.current_model_loaded = "Chatterbox TTS"
            print(f"[{self.branding}ENG] \033[92mChatterbox TTS model loaded successfully on {self.device}\033[0m")
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError loading Chatterbox TTS model: {str(e)}\033[0m")
            self.is_tts_model_loaded = False
            raise e

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
        if self.lowvram_capable and self.cuda_is_available:
            if self.lowvram_enabled:
                # Move model to CPU
                if self.model is not None and self.device == "cuda":
                    print(f"[{self.branding}ENG] \033[93mMoving model to CPU (Low VRAM mode)\033[0m")
                    self.model.to('cpu')
                    torch.cuda.empty_cache()
            else:
                # Move model back to CUDA
                if self.model is not None and self.device == "cuda":
                    print(f"[{self.branding}ENG] \033[93mMoving model to CUDA\033[0m")
                    self.model.to('cuda')
        
    ########################################
    ########################################
    # CHANGE ME #  DeepSpeed model loading #
    ########################################
    ########################################
    async def handle_deepspeed_change(self, value):
        # Chatterbox TTS doesn't support DeepSpeed
        print(f"[{self.branding}ENG] \033[93mChatterbox TTS does not support DeepSpeed\033[0m")

    def set_seed(self, seed: int):
        """Set seed for reproducible generation across PyTorch, CUDA, and NumPy."""
        # If seed is 0, generate a random seed
        if seed == 0:
            seed = random.randint(1, 2**31 - 1)
            print(f"[{self.branding}ENG] \033[93mGenerated random seed: {seed}\033[0m") if self.debug_tts else None
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"[{self.branding}ENG] \033[93mSeed set to: {seed}\033[0m") if self.debug_tts else None
        
    def scan_models_folder(self):
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # Chatterbox TTS uses auto-downloaded models, return as dictionary for API compatibility
        available_models = {
            "Chatterbox TTS": {"model_name": "Chatterbox TTS", "folder_path": "chatterbox-default"}
        }
        
        # Set self.available_models for API access
        self.available_models = available_models
        
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 
        return available_models
        
    def voices_file_list(self):       
        try:
            voices = []
            directory = self.main_dir / "voices"
            
            # Step 1: Add .wav files in the main "voices" directory to the list
            for f in directory.glob("*.wav"):
                voices.append(f.name)
            
            # Step 2: Walk through subfolders and add subfolder names if they contain wav files
            for folder in directory.iterdir():
                if folder.is_dir():
                    has_wav_files = any(folder.glob("*.wav"))
                    if has_wav_files:
                        folder_name = folder.name + "/"
                        voices.append(folder_name)
            
            # Remove "voices/" from the list if it somehow got added
            voices = [v for v in voices if v != "voices/"]
                        
            if not voices:
                return ["No Voices Found"] 
            return voices 
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Voices/Voice Models not found. Cannot load a list of voices.")
            print(f"[{self.branding}ENG]")
            return ["No Voices Found"]

    async def api_manual_load_model(self, model_name):
        if model_name == "Chatterbox TTS":
            if not self.is_tts_model_loaded:
                await self.setup()
            return f"Chatterbox TTS model loaded: {model_name}"
        else:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not found")

    async def unload_model(self):
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        
        if self.model is not None:
            print(f"[{self.branding}ENG] \033[93mUnloading Chatterbox TTS model\033[0m")
            del self.model
            self.model = None
            self.is_tts_model_loaded = False
            self.current_model_loaded = None
            if self.cuda_is_available:
                torch.cuda.empty_cache()
        
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 

    async def handle_tts_method_change(self, tts_method):
        # Chatterbox TTS is simple and doesn't need method changes
        pass

    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming, seed=None, exaggeration=0.5, cfg_weight=0.5, min_p=0.05, top_p=1.0):
        if voice == "No Voices Found":
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No voices found to generate TTS.")
            raise HTTPException(status_code=400, detail="No voices found to generate TTS.")
            
        if not self.is_tts_model_loaded:
            raise HTTPException(status_code=500, detail="Chatterbox TTS model not loaded")
            
        if self.tts_generating_lock:
            raise HTTPException(status_code=503, detail="TTS generation already in progress")
        
        self.tts_generating_lock = True
        generate_start_time = time.time()
        
        try:
            print(f"[{self.branding}ENG] \033[92mGenerating TTS with Chatterbox...\033[0m")
            
            # Handle low VRAM if enabled
            if self.lowvram_enabled and self.cuda_is_available:
                self.model.to(self.device)
            
            # Check for stop generation flag
            if self.tts_stop_generation:
                self.tts_stop_generation = False
                self.tts_generating_lock = False
                return
            
            # Determine the audio prompt path
            audio_prompt_path = None
            if voice and voice != "No Voices Found":
                if voice.endswith('/'):
                    # It's a folder, pick the first wav file
                    voice_dir = self.main_dir / "voices" / voice.rstrip('/')
                    for wav_file in voice_dir.glob("*.wav"):
                        audio_prompt_path = str(wav_file)
                        break
                else:
                    # It's a direct file
                    audio_prompt_path = str(self.main_dir / "voices" / voice)
                
                # Verify the file exists
                if audio_prompt_path and not os.path.exists(audio_prompt_path):
                    print(f"[{self.branding}ENG] \033[91mWarning: Voice file not found: {audio_prompt_path}\033[0m")
                    audio_prompt_path = None
            
            # Map AllTalk parameters to Chatterbox parameters
            chatterbox_params = {
                'text': text,
                'temperature': float(temperature),
                'repetition_penalty': float(repetition_penalty),
                'exaggeration': float(exaggeration),
                'cfg_weight': float(cfg_weight),
                'min_p': float(min_p),
                'top_p': float(top_p),
            }
            
            # Set seed for reproducible generation if provided
            if seed is not None:
                self.set_seed(int(seed))
            
            # Add audio prompt if available
            if audio_prompt_path:
                chatterbox_params['audio_prompt_path'] = audio_prompt_path
                print(f"[{self.branding}ENG] Using voice prompt: {audio_prompt_path}") if self.debug_tts else None
            
            print(f"[{self.branding}ENG] Generation parameters: temp={temperature}, rep_penalty={repetition_penalty}, exag={exaggeration}, cfg={cfg_weight}, min_p={min_p}, top_p={top_p}, seed={seed}") if self.debug_tts else None
            
            # Generate TTS
            wav = self.model.generate(**chatterbox_params)
            
            # Check for stop generation flag again
            if self.tts_stop_generation:
                self.tts_stop_generation = False
                self.tts_generating_lock = False
                return
            
            # Save the generated audio
            ta.save(output_file, wav, self.model.sr)
            
            generate_end_time = time.time()
            generate_elapsed_time = generate_end_time - generate_start_time
            print(f"[{self.branding}GEN] \033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{self.lowvram_enabled} \033[94mDeepSpeed: \033[33m{self.deepspeed_enabled}\033[0m")
            
            # Handle low VRAM cleanup
            if self.lowvram_enabled and self.cuda_is_available and not self.tts_narrator_generatingtts:
                self.model.to('cpu')
                torch.cuda.empty_cache()
            
            # Handle streaming if requested
            if streaming:
                with open(output_file, 'rb') as f:
                    yield f.read()
                
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError during TTS generation: {str(e)}\033[0m")
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
        finally:
            self.tts_generating_lock = False 