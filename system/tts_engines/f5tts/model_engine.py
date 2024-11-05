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
# In this section you will import any imports that your specific TTS Engine will use. You will provide any
# start-up errors for those bits, as if you were starting up a normal Python script. Note the logging.disable
# a few lines up from here, you may want to # that out while debugging!
from vocos import Vocos
import soundfile as sf
import tempfile
from pydub import AudioSegment, silence
import re
import torchaudio
import numpy as np
import subprocess
import sys

def install_and_restart():
    try:
        print("##########################################")
        print("F5-TTS not found. Attempting to install...")
        print("##########################################")
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "git+https://github.com/SWivid/F5-TTS.git"
        ])
        print("########################################################")
        print("F5-TTS installed successfully! Restarting application...")
        print("########################################################")
        
        # Get the current script's path
        script_path = sys.argv[0]
        
        # Restart the current script
        os.execv(sys.executable, ['python'] + sys.argv)
        
    except subprocess.CalledProcessError as e:
        print("########################################################")
        print(f"Failed to install F5-TTS: {str(e)}")
        print("########################################################")
        raise ImportError("Could not install required package F5-TTS")

try:
    from f5_tts.model import CFM, DiT, UNetT
    from f5_tts.model.utils import (
        get_tokenizer,
        convert_char_to_pinyin,
    )
except ImportError:
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
        # DONT CHANGE #  These settings are specific to the F5-TTS Model/Engine ####
        ############################################################################
        # Add F5-TTS specific parameters
        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.win_length = 1024
        self.n_fft = 1024
        self.mel_spec_type = 'vocos'
        self.target_rms = 0.1
        self.cross_fade_duration = 0.15
        self.ode_method = 'euler'
        self.nfe_step = 32
        self.cfg_strength = 2.0
        self.sway_sampling_coef = -1.0
        self.speed = 1.0
        # F5-TTS model configuration
        self.f5_model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4
        )
        # Add E2-TTS model configuration
        self.e2_model_cfg = dict(
            dim=1024,
            depth=24,
            heads=16,
            ff_mult=4
        )       
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
    # In here you will add code to load in your model and do its inital setup. So you
    # may be calling your model loader via handle_tts_method_change or if your TTS
    # engine doesnt actually load a model into CUDA or System RAM, you may be doing 
    # Something to fake its start-up.
    async def setup(self):
        self.printout_versions()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # Record the start time of loading the model
        generate_start_time = time.time()
        
        # Scan for available models
        self.available_models = self.scan_models_folder()
        print("in setup - self.selected_model:", self.selected_model) if self.debug_tts else None
        
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
        
        print(f"[{self.branding}ENG]\033[94m Model/Engine :\033[93m F5-TTS\033[94m Ready\033[0m")
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
    # If your model does load into CUDA and you want to support LowVRAM, aka moving the model
    # on the fly between CUDA and System RAM on each generation request, you will be adding
    # The code in here to do it. See the XTTS tts engine for an example. Piper however, doesnt
    # Load into VRAM or System RAM, also low vram is set globally disabled by the model_settings.JSON
    # file, so this would never get called anywyay, however, we still need to keep an empty 
    # function in place. The "pass" tells the function to just exit out cleanly if called.
    # However, its quite a simple check along the lines of "if CUDA is available and model is
    # in X place, then send it to Y place (or Y to X).
    async def handle_lowvram_change(self):
        """Handle moving model between CPU and GPU for low VRAM operation"""
        if not self.lowvram_capable or not self.cuda_is_available:
            return

        if not self.is_tts_model_loaded:
            return

        try:
            if self.lowvram_enabled and self.device == 'cuda':
                # Moving to CPU - everything must be float32
                print(f"[{self.branding}Debug] Moving models to CPU") if self.debug_tts else None
                
                if hasattr(self, 'model'):
                    self.model = self.model.float().to('cpu')
                    torch.cuda.empty_cache()
                    print(f"[{self.branding}Debug] Model moved to CPU") if self.debug_tts else None
                
                if hasattr(self, 'vocoder'):
                    # Move vocoder without checking device
                    try:
                        self.vocoder = self.vocoder.float().to('cpu')
                        torch.cuda.empty_cache()
                        print(f"[{self.branding}Debug] Vocoder moved to CPU") if self.debug_tts else None
                    except Exception as e:
                        print(f"[{self.branding}Debug] Vocoder move warning: {str(e)}") if self.debug_tts else None
                
                self.device = 'cpu'
                print(f"[{self.branding}ENG] Models moved to CPU for low VRAM mode") if self.debug_tts else None
                
            elif self.lowvram_enabled and self.device == 'cpu':
                # Moving to GPU - everything must be float16
                print(f"[{self.branding}Debug] Moving models to GPU") if self.debug_tts else None
                
                if hasattr(self, 'model'):
                    self.model = self.model.to('cuda').half()
                    print(f"[{self.branding}Debug] Model moved to GPU") if self.debug_tts else None
                
                if hasattr(self, 'vocoder'):
                    try:
                        self.vocoder = self.vocoder.to('cuda').half()
                        print(f"[{self.branding}Debug] Vocoder moved to GPU") if self.debug_tts else None
                    except Exception as e:
                        print(f"[{self.branding}Debug] Vocoder move warning: {str(e)}") if self.debug_tts else None
                    
                self.device = 'cuda'
                print(f"[{self.branding}ENG] Models moved to GPU") if self.debug_tts else None
            
            # Verify movements and precision
            if self.debug_tts:
                if hasattr(self, 'model'):
                    # Check model parameters only
                    params_dtype = next(self.model.parameters()).dtype
                    print(f"[{self.branding}Debug] Model parameters dtype: {params_dtype}")
                    print(f"[{self.branding}Debug] Final model device: {next(self.model.parameters()).device}")
                
                if hasattr(self, 'vocoder'):
                    try:
                        # Try to check vocoder parameters
                        params = next(self.vocoder.parameters())
                        print(f"[{self.branding}Debug] Vocoder parameters dtype: {params.dtype}")
                    except Exception:
                        pass  # Ignore vocoder parameter checking errors
                    
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
    # If the model supports CUDA and DeepSpeed, this is where you can handle re-loading
    # the model as/when DeepSpeed is enabled/selected in the user interface. If your 
    # TTS model doesnt support DeepSpeed, then it should be globally set in your
    # model_settings.JSON and this function will never get called, however it still needs
    # to exist as a function.
    async def handle_deepspeed_change(self, value):
        """F5-TTS currently doesn't support DeepSpeed"""
        print(f"[{self.branding}ENG] DeepSpeed not supported for F5-TTS")
        return False


    ##############################################################################################################################################
    ##############################################################################################################################################
    # CHANGE ME # scan for available models/voices that are relevant to this TTS engine # XTTS is very unusal as it has 2x model loading methods #
    ##############################################################################################################################################
    ##############################################################################################################################################
    # This function looks and reports back the list of possible models your TTS engine can
    # load in. Some TTS engines have multiple models they can load and you will want to use
    # code for checking if the models are in the correct location/placement within the disk
    # the correct files exst per model etc (check XTTS for an example of this). Some models
    # like Piper, the actual models are the voices, so in the Piper scan_models_folder
    # function, we fake the only model being available as {'piper': 'piper'} aka, model name
    # then engine name, then we use the voices_file_list to populate the models as available
    # voices that can be selected in the interface.    
    # If no models are found, we return "No Models Available" and continue on with the script.
    def scan_models_folder(self):
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        
        """Scan for available F5-TTS models"""
        self.available_models = {}
        models_dir = self.main_dir / "models" / "f5tts"
        
        if not models_dir.exists():
            print(f"[{self.branding}ENG] Models directory not found: {models_dir}")
            self.available_models["No Models Found"] = "No Models Found"
            return self.available_models
            
        # Look for model directories that match the pattern f5tts_v*
        for model_dir in models_dir.glob("*tts_v*"):
            if model_dir.is_dir():
                # First try to find model_*.safetensors files
                model_files = list(model_dir.glob("model_*.safetensors"))
                if not model_files:
                    # If no model_*.safetensors found, try any .safetensors file
                    model_files = list(model_dir.glob("*.safetensors"))
                    
                vocab_file = model_dir / "vocab.txt"
                vocos_dir = model_dir / "vocos"
                vocos_config = vocos_dir / "config.yaml"
                vocos_model = vocos_dir / "pytorch_model.bin"
                
                # Check if we have at least one model file and all other required files
                if model_files and all(f.exists() for f in [vocab_file, vocos_config, vocos_model]):
                    model_name = model_dir.name
                    self.available_models[model_name] = model_name
        
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 
        if not self.available_models:
            self.available_models["No Models Found"] = "No Models Found" # Return a list with {'No Models Found'} if there are no models found.
        return self.available_models # Return a list of models in the format {'engine name': 'model 1', 'engine name': 'model 2", etc}

    #############################################################
    #############################################################
    # CHANGE ME #  POPULATE FILES LIST FROM VOICES DIRECTORY ####
    #############################################################
    #############################################################
    # This function looks and reports back the list of possible voice your TTS engine can
    # load in. Some TTS engines the voices are wav file samples (XTTS), some are models 
    # (Piper) and some are text (Parler) thats stored in a JSON file. We just need to 
    # populate the "voices" variable somehow and if no voices are found, we return
    # "No Voices Found" back to the interface/api. 
    def voices_file_list(self):
        try:
            voices = []
            # Function to check if a wav file has a corresponding reference text file
            def has_reference_text(wav_path):
                text_path = wav_path.with_suffix('.reference.txt')
                return text_path.exists()
            
            directory = self.main_dir / "voices"
            
            # Step 1: Add .wav files in the main "voices" directory to the list (only if they have matching .reference.txt)
            for f in directory.glob("*.wav"):
                if has_reference_text(f):
                    voices.append(f.name)
                else:
                    print(f"[{self.branding}ENG] Warning: {f.name} does not have a matching reference text file") if self.debug_tts else None
            
            # Step 2: Walk through subfolders and add subfolder names if they contain valid wav+reference.txt pairs
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
    # This function will handle the loading of your model, into VRAM/CUDA, System RAM or whatever.
    async def api_manual_load_model(self, model_name):
        if model_name == "No Models Found":
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load.")
            raise HTTPException(status_code=400, detail="No models for this TTS engine were found to load.")
            
        print(f"[{self.branding}ENG]\033[94m Model/Engine :\033[93m {model_name}\033[94m loading into\033[93m", self.device,"\033[0m")
        
        # Split the engine name from model name and get just the model folder name
        model_folder = model_name.split(" - ")[-1]
        
        try:
            # Set up paths using the correct model folder name
            model_dir = self.main_dir / "models" / "f5tts" / model_folder
            vocab_path = model_dir / "vocab.txt"
            vocos_path = model_dir / "vocos"
            
            # Dynamically find the safetensors model file
            model_files = list(model_dir.glob("model_*.safetensors"))
            if not model_files:
                # Try finding any safetensors file as fallback
                model_files = list(model_dir.glob("*.safetensors"))
                
            if not model_files:
                raise FileNotFoundError("No safetensors model file found in the model directory")
                
            # Use the first found model file
            model_path = model_files[0]
            
            print(f"[{self.branding}ENG] Loading model from path: {model_path}") if self.debug_tts else None
            
            # Initialize vocoder - always in float32 initially
            self.vocoder = Vocos.from_hparams(str(vocos_path / "config.yaml"))
            vocoder_state = torch.load(str(vocos_path / "pytorch_model.bin"), map_location="cpu")
            self.vocoder.load_state_dict(vocoder_state)
            self.vocoder = self.vocoder.eval()
            
            # Determine if this is an E2-TTS model from the folder name
            is_e2_model = "e2tts" in model_folder.lower()
            
            # Initialize model with appropriate architecture
            vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
            
            mel_spec_kwargs = dict(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mel_channels=self.n_mel_channels,
                target_sample_rate=self.target_sample_rate,
                mel_spec_type=self.mel_spec_type,
            )
            
            if is_e2_model:
                print(f"[{self.branding}ENG] Initializing E2-TTS model") if self.debug_tts else None
                self.model = CFM(
                    transformer=UNetT(
                        **self.e2_model_cfg,
                        text_num_embeds=vocab_size,
                        mel_dim=self.n_mel_channels
                    ),
                    mel_spec_kwargs=mel_spec_kwargs,
                    odeint_kwargs=dict(
                        method=self.ode_method,
                    ),
                    vocab_char_map=vocab_char_map
                ).float()  # Ensure float32
            else:
                print(f"[{self.branding}ENG] Initializing F5-TTS model") if self.debug_tts else None
                self.model = CFM(
                    transformer=DiT(
                        **self.f5_model_cfg,
                        text_num_embeds=vocab_size,
                        mel_dim=self.n_mel_channels
                    ),
                    mel_spec_kwargs=mel_spec_kwargs,
                    odeint_kwargs=dict(
                        method=self.ode_method,
                    ),
                    vocab_char_map=vocab_char_map
                ).float()  # Ensure float32
            
            # Load model weights
            if str(model_path).endswith('.safetensors'):
                from safetensors.torch import load_file
                checkpoint = load_file(model_path)
                checkpoint = {"ema_model_state_dict": checkpoint}
            else:
                checkpoint = torch.load(model_path, weights_only=True)
                
            # Handle backward compatibility before loading weights
            model_state_dict = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            
            # Remove backward compatibility keys before loading
            backward_compat_keys = [
                "mel_spec.mel_stft.mel_scale.fb",
                "mel_spec.mel_stft.spectrogram.window"
            ]
            for key in backward_compat_keys:
                if key in model_state_dict:
                    del model_state_dict[key]
            
            self.model.load_state_dict(model_state_dict)
            self.model.eval()
            
            # Move to device and handle precision
            if self.device == "cuda":
                self.model = self.model.to(self.device).half()
                self.vocoder = self.vocoder.to(self.device).half()
                
                # Ensure all parts are in half precision
                for module in [self.model, self.vocoder]:
                    for param in module.parameters():
                        param.data = param.data.half()
                    for buffer in module.buffers():
                        if buffer.dtype == torch.float32:
                            buffer.data = buffer.data.half()
            else:
                self.model = self.model.to(self.device)
                self.vocoder = self.vocoder.to(self.device)
                
            self.is_tts_model_loaded = True
            model_type = "E2-TTS" if is_e2_model else "F5-TTS"
            print(f"[{self.branding}ENG] {model_type} model loaded successfully") if self.debug_tts else None
            
        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError loading model: {str(e)}\033[0m")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
            
    ###############################
    ###############################
    # CHANGE ME # Model unloading #
    ###############################
    ###############################
    # This function will handle the UN-loading of your model, from VRAM/CUDA, System RAM or whatever.
    async def unload_model(self):
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        
        """Unload the F5-TTS model from memory"""
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        
        if hasattr(self, 'vocoder'):
            del self.vocoder
            self.vocoder = None
            
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
    # This function is your central model loading/unloading handler that deals with the above functions as necesary, to call loading, unloading,
    # swappng DeepSpeed, Low vram etc. This function gets called with a "engine name - model name" type call.
    async def handle_tts_method_change(self, tts_method):
        generate_start_time = time.time() # Record the start time of loading the model
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
        generate_end_time = time.time() # Create an end timer for calculating load times
        generate_elapsed_time = generate_end_time - generate_start_time  # Calculate start time minus end time
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m") # Print out the result of the load time
        return True


    async def preprocess_ref_audio_text(self, ref_audio_orig, ref_text, clip_short=True):
        print(f"[{self.branding}ENG] Converting audio...") if self.debug_tts else None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            aseg = AudioSegment.from_file(ref_audio_orig)

            if clip_short:
                # 1. try to find long silence for clipping
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                        print(f"[{self.branding}ENG] Audio is over 15s, clipping short. (1)") if self.debug_tts else None
                        break
                    non_silent_wave += non_silent_seg

                # 2. try to find short silence for clipping if 1. failed
                if len(non_silent_wave) > 15000:
                    non_silent_segs = silence.split_on_silence(
                        aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000
                    )
                    non_silent_wave = AudioSegment.silent(duration=0)
                    for non_silent_seg in non_silent_segs:
                        if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                            print(f"[{self.branding}ENG] Audio is over 15s, clipping short. (2)") if self.debug_tts else None
                            break
                        non_silent_wave += non_silent_seg

                aseg = non_silent_wave

                # 3. if no proper silence found for clipping
                if len(aseg) > 15000:
                    aseg = aseg[:15000]
                    print(f"[{self.branding}ENG] Audio is over 15s, clipping short. (3)") if self.debug_tts else None

            aseg.export(f.name, format="wav")
            ref_audio = f.name

        # Ensure ref_text ends with a proper sentence-ending punctuation
        if not ref_text.endswith(". ") and not ref_text.endswith("。"):
            if ref_text.endswith("."):
                ref_text += " "
            else:
                ref_text += ". "

        return ref_audio, ref_text

    async def infer_process(
        self,
        ref_audio,
        ref_text,
        gen_text,
        model_obj,
        vocoder,
        target_rms=0.1,
        cross_fade_duration=0.15,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1,
        speed=1,
        fix_duration=None,
        device=None
    ):
        """Process text and prepare for batch inference"""
        # Split the input text into batches
        audio, sr = torchaudio.load(ref_audio)
        max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
        gen_text_batches = self.chunk_text(gen_text, max_chars=max_chars)
        
        for i, gen_text_batch in enumerate(gen_text_batches):
            print(f"[{self.branding}ENG] gen_text {i}", gen_text_batch) if self.debug_tts else None

        print(f"[{self.branding}ENG] Generating audio in {len(gen_text_batches)} batches...") if self.debug_tts else None
        
        return await self.infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device
        )

    async def infer_batch_process(
        self,
        ref_audio,
        ref_text,
        gen_text_batches,
        model_obj,
        vocoder,
        target_rms=0.1,
        cross_fade_duration=0.15,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1,
        speed=1,
        fix_duration=None,
        device=None
    ):
        """Process batches for inference"""
        device = device or self.device
        audio, sr = ref_audio
        
        # Always process initial audio in float32
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        audio = audio.to(torch.float32)  # Ensure float32 for initial processing
        
        # Normalize audio if needed
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        
        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)

        # Move to device and adjust precision
        audio = audio.to(device)
        if device == 'cuda':
            audio = audio.half()

        generated_waves = []
        spectrograms = []

        if len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "
            
        for gen_text in gen_text_batches:
            # Prepare the text
            text_list = [ref_text + gen_text]
            final_text_list = convert_char_to_pinyin(text_list)

            ref_audio_len = audio.shape[-1] // self.hop_length
            if fix_duration is not None:
                duration = int(fix_duration * self.target_sample_rate / self.hop_length)
            else:
                # Calculate duration
                ref_text_len = len(ref_text.encode("utf-8"))
                gen_text_len = len(gen_text.encode("utf-8"))
                duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

            # inference
            with torch.inference_mode():
                generated, _ = model_obj.sample(
                    cond=audio,
                    text=final_text_list,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )

                # Handle precision for post-processing
                if self.device == 'cuda':
                    generated = generated.half()
                else:
                    generated = generated.float()

                generated = generated[:, ref_audio_len:, :]
                generated_mel_spec = generated.permute(0, 2, 1)
                
                # Convert to the correct precision for vocoder
                if self.device == 'cuda':
                    generated_mel_spec = generated_mel_spec.half()
                else:
                    generated_mel_spec = generated_mel_spec.float()
                    
                generated_wave = vocoder.decode(generated_mel_spec)
                if rms < target_rms:
                    generated_wave = generated_wave * rms / target_rms

                # wav -> numpy
                generated_wave = generated_wave.squeeze().cpu().float().numpy()  # Always convert to float32 for numpy

                generated_waves.append(generated_wave)
                spectrograms.append(generated_mel_spec[0].cpu().numpy())

        # Combine all generated waves with cross-fading
        if cross_fade_duration <= 0:
            final_wave = np.concatenate(generated_waves)
        else:
            final_wave = generated_waves[0]
            for i in range(1, len(generated_waves)):
                prev_wave = final_wave
                next_wave = generated_waves[i]

                # Calculate cross-fade samples
                cross_fade_samples = int(cross_fade_duration * self.target_sample_rate)
                cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                if cross_fade_samples <= 0:
                    final_wave = np.concatenate([prev_wave, next_wave])
                    continue

                # Overlapping parts
                prev_overlap = prev_wave[-cross_fade_samples:]
                next_overlap = next_wave[:cross_fade_samples]

                # Fade out and fade in
                fade_out = np.linspace(1, 0, cross_fade_samples)
                fade_in = np.linspace(0, 1, cross_fade_samples)

                # Cross-faded overlap
                cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                # Combine
                new_wave = np.concatenate(
                    [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                )

                final_wave = new_wave

        # Create a combined spectrogram
        combined_spectrogram = np.concatenate(spectrograms, axis=1)

        return final_wave, self.target_sample_rate, combined_spectrogram

    def chunk_text(self, text, max_chars=135):
        """
        Splits the input text into chunks, each with a maximum number of characters.
        """
        chunks = []
        current_chunk = ""
        # Split the text into sentences based on punctuation followed by whitespace
        sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

        for sentence in sentences:
            if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
                current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    #############################################################################################################################
    #############################################################################################################################
    # DONT CHANGE ME # De-Capitalise words (Not used currently) #################################################################
    #############################################################################################################################
    #############################################################################################################################
    def process_text_for_tts(self, text):
        """
        Process text by converting fully capitalized words to lowercase,
        while preserving other words' capitalization.
        
        Args:
            self: The class instance
            text (str): Input text string
            
        Returns:
            str: Processed text with fully capitalized words converted to lowercase
        """
        # Split text into words
        words = text.split()
        
        # Process each word
        processed_words = []
        for word in words:
            # Check if word is fully uppercase and longer than 1 character
            if word.isupper() and len(word) > 1:
                processed_words.append(word.lower())
            else:
                processed_words.append(word)
        
        # Join words back together
        return ' '.join(processed_words)

    ##########################################################################################################################################
    ##########################################################################################################################################
    # CHANGE ME # Model changing. Unload out old model and load in a new one #################################################################
    ##########################################################################################################################################
    ##########################################################################################################################################
    # In here all the possible options are sent over (text, voice to use, lanugage, speed etc etc) and its up to you how you use them, or not.
    # obviously if your TTS engine doesnt support speed for example, generationspeed_capable should be set False in your model_settings.JSON file
    # and a fake "generationspeed_set" value should be set. This allows a fake value to be sent over from the main script, even though it
    # wouldnt actually ever be used in the generation below. Nonethless all these values, real or just whats inside the configuration file
    # will be sent over for use. 
    # Setting the xxxxxxx_capabale in the model_settings.JSON file, will enable/disable them being selectable by the user. For example, if you
    # set "generationspeed_capable" as false in the model_settings.JSON file, a user will not be able to select OR set the setting for 
    # generation speed.
    # One thing to note is that we HAVE to have something in this generation request that is synchronous from the way its called, which means
    # we have to have an option for Streaming, even if our TTS engine doesnt support streaming. So in that case, we would set streaming_capable
    # as false in our model_settings.JSON file, meaning streaming will never be called. However, we have to put a fake streaming routine in our
    # function below (or a real function if it does support streaming of course). Parler has an example of a fake streaming function, which is
    # very clearly highlighted in its model_engine.py script.   
    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming):
        if voice == "No Voices Found":
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No voices found to generate TTS.")
            raise HTTPException(status_code=400, detail="No voices found to generate TTS.")    
        if not self.is_tts_model_loaded:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: You currently have no TTS model loaded.")
            raise HTTPException(status_code=400, detail="You currently have no TTS model loaded.")
        
        # Language validation - converted to lowercase for case-insensitive comparison
        language = language.lower()
        if language not in ['en', 'eng', 'zh', 'zho', 'chi']:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: F5-TTS only supports English (EN) or Chinese (ZH) language TTS generation.")
            print(f"[{self.branding}ENG] \033[91mError\033[0m: You may get an error message or gibberish output.")
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Please use only EN or ZH/ZHO for the language with F5-TTS.") 
            
        self.tts_generating_lock = True
        generate_start_time = time.time()

        try:
            if self.lowvram_enabled and self.device == "cpu":
                print(f"[{self.branding}Debug] Moving models to GPU for generation") if self.debug_tts else None
                await self.handle_lowvram_change()
                
                # Verify precision is consistent
                if self.debug_tts:
                    print(f"[{self.branding}Debug] Verifying model precision...")
                    if hasattr(self, 'model'):
                        params_dtype = {p.dtype for p in self.model.parameters()}
                        buffers_dtype = {b.dtype for b in self.model.buffers()}
                        if len(params_dtype) > 1 or len(buffers_dtype) > 1:
                            print(f"[{self.branding}ENG] Warning: Mixed precision detected in model")
                            print(f"[{self.branding}Debug] Model parameters dtypes: {params_dtype}")
                            print(f"[{self.branding}Debug] Model buffers dtypes: {buffers_dtype}")
                            
                    if hasattr(self, 'vocoder'):
                        params_dtype = {p.dtype for p in self.vocoder.parameters()}
                        buffers_dtype = {b.dtype for b in self.vocoder.buffers()}
                        if len(params_dtype) > 1 or len(buffers_dtype) > 1:
                            print(f"[{self.branding}ENG] Warning: Mixed precision detected in vocoder")
                            print(f"[{self.branding}Debug] Vocoder parameters dtypes: {params_dtype}")
                            print(f"[{self.branding}Debug] Vocoder buffers dtypes: {buffers_dtype}")
            
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
                
            # Process reference audio and text
            ref_audio, processed_ref_text = await self.preprocess_ref_audio_text(
                str(ref_audio_path), ref_text
            )
            
            text = self.process_text_for_tts(text)
            
            # Generate the audio using infer_process
            final_wave, final_sample_rate, _ = await self.infer_process(
                ref_audio,
                processed_ref_text,
                text,
                self.model,
                self.vocoder,
                target_rms=self.target_rms,
                cross_fade_duration=self.cross_fade_duration,
                nfe_step=self.nfe_step,
                cfg_strength=self.cfg_strength,
                sway_sampling_coef=self.sway_sampling_coef,
                speed=float(speed),
                fix_duration=None,
                device=self.device
            )
            
            # Save the audio
            sf.write(output_file, final_wave, final_sample_rate)
            
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
            raise HTTPException(status_code=500, detail=f"Error during TTS generation: {str(e)}")
        finally:
            try:
                # Handle low VRAM mode - move back to CPU if needed
                if self.lowvram_enabled and not self.tts_narrator_generatingtts:
                    print(f"[{self.branding}Debug] Moving models back to CPU after generation") if self.debug_tts else None
                    await self.handle_lowvram_change()
                
                # Verify final device state in debug mode using parameter checking
                if self.debug_tts:
                    if hasattr(self, 'model'):
                        model_device = next(self.model.parameters()).device
                        print(f"[{self.branding}Debug] Final model device: {model_device}") if self.debug_tts else None
                    if hasattr(self, 'vocoder'):
                        try:
                            vocoder_device = next(self.vocoder.parameters()).device
                            print(f"[{self.branding}Debug] Final vocoder device: {vocoder_device}") if self.debug_tts else None
                        except Exception:
                            # Skip vocoder device checking if it fails
                            pass
            except Exception as e:
                print(f"[{self.branding}Debug] Device verification warning: {str(e)}") if self.debug_tts else None
            
            self.tts_generating_lock = False