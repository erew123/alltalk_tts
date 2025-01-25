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
# In this section you will import any imports that your specific TTS Engine will use. You will provide any
# start-up errors for those bits, as if you were starting up a normal Python script. Note the logging.disable
# a few lines up from here, you may want to # that out while debugging!

import torchaudio
import soundfile as sf
import tempfile
from pydub import AudioSegment, silence
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import soundfile as sf
from xcodec2.modeling_xcodec2 import XCodec2Model
from . import llasa


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

        #####


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

        self.available_models = self.scan_models_folder()
        if self.selected_model:
            tts_model = f"{self.selected_model}"
            if tts_model in self.available_models:
                await self.handle_tts_method_change(tts_model)
                self.current_model_loaded = tts_model
            else:
                self.current_model_loaded = "No Models Available"
                print(f"[{self.branding}ENG] \033[91mError\033[0m: Selected model '{self.selected_model}' not found in the models folder.")

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
        pass # Piper does not stay in CUDA or support swapping of location

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
        if value:
            # DeepSpeed enabled
            print(f"[{self.branding}ENG] \033[93mDeepSpeed Activating\033[0m")
            await self.unload_model()
            self.params["tts_method_api_local"] = True
            self.deepspeed_enabled = True
            await self.setup()
        else:
            # DeepSpeed disabled
            print(f"[{self.branding}ENG] \033[93mDeepSpeed De-Activating\033[0m")
            self.deepspeed_enabled = False
            await self.unload_model()
            await self.setup()
        return value  # Return new checkbox value

    #####################################################################################
    #####################################################################################
    # CHANGE ME # scan for available models/voices that are relevant to this TTS engine #
    #####################################################################################
    #####################################################################################
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
        models_folder = self.main_dir / "models" / "llasa"
        print("models_folder is:", models_folder) if self.debug_tts else None
        self.available_models = {}

        # Add immediate check for models folder existence
        if not models_folder.exists():
            print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Models folder does not exist: {models_folder}")
            print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Please use the Gradio inteface to download/select a model.")
            self.available_models = {'No Models Available': 'llasa'}
            return self.available_models

        # Check basic files needed for any model
        common_required_files = [
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json"
        ]

        found_valid_model = False
        for subfolder in models_folder.iterdir():
            if subfolder.is_dir():
                model_name = subfolder.name
                print("model_name is:", model_name) if self.debug_tts else None

                # Check for common required files
                if all(subfolder.joinpath(file).exists() for file in common_required_files):
                    # Look for any safetensors files
                    safetensor_files = list(subfolder.glob("*.safetensors"))
                    if safetensor_files:
                        self.available_models[f"llasa - {model_name}"] = "llasa"
                        found_valid_model = True
                        print("self.available_models is:", self.available_models) if self.debug_tts else None
                    else:
                        print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Model folder '{model_name}' has no .safetensors files.")
                else:
                    print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Model folder '{model_name}' is missing common required files.")

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # ↓↓↓ Keep everything below this line ↓↓↓
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

        if not found_valid_model:
            self.available_models = {'No Models Available': 'llasa'}
            print(f"[{self.branding}ENG] \033[91mWarning\033[0m: No valid llasa models found.")
            print(f"[{self.branding}ENG] \033[91mWarning\033[0m: Please download and install llasa models.")

        return self.available_models

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
    # CHANGE ME # Model loading # Piper does not actually load/stay resident in RAM #
    #################################################################################
    #################################################################################
    # This function will handle the loading of your model, into VRAM/CUDA, System RAM or whatever.
    # In XTTS, which has 2x model loader types, there are 2x loaders. They are called by "def handle_tts_method_change"
    # In Piper we fake a model loader as Piper doesnt actually load a model into CUDA/System RAM as such. So, in that
    # situation, api_manual_load_model is kind of a blank function. Though we do set self.is_tts_model_loaded = True
    # as this is used elsewhere in the scripts to confirm that a model is available to be used for TTS generation.
    # We always check for "No Models Available" being sent as that means we are trying to load in a model that
    # doesnt exist/wasnt found on script start-up e.g. someone deleted the model from the folder or something.
    async def api_manual_load_model(self, model_name):
        if model_name == "No Models Found":
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load.")
            raise HTTPException(status_code=400, detail="No models for this TTS engine were found to load.")
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        model_path = self.main_dir / "models" / "llasa" / model_name
        print("model_path is:", model_path) if self.debug_tts_variables else None

        if not model_path.exists():
            print(f"[{self.branding}ENG] \033[91mError\033[0m: Model directory not found: {model_path}")
            print(f"[{self.branding}ENG] \033[93mPlease download a llasa model file in the Gradio interface section for the llasa engine.\033[0m")
            raise HTTPException(status_code=404, detail=f"Model directory not found: {model_path}")

        try:
            model = AutoModelForCausalLM.from_pretrained('srinivasbilla/Llasa-3B')
            model.eval()
            model.to(self.device)
            codec = XCodec2Model.from_pretrained("HKUST-Audio/xcodec2").eval().to(self.device)
            self.model = (model, codec)
            self.tokenizer = AutoTokenizer.from_pretrained('srinivasbilla/Llasa-3B')

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**2
                print(f"[{self.branding}ENG] \033[94mGPU Memory Used:\033[93m {memory_used:.2f} MB\033[0m")

        except Exception as e:
            print(f"[{self.branding}ENG] \033[91mError loading model:\033[0m {str(e)}")
            return None

        self.is_tts_model_loaded = True
        return self.model

    ###############################
    ###############################
    # CHANGE ME # Model unloading #
    ###############################
    ###############################
    # This function will handle the UN-loading of your model, from VRAM/CUDA, System RAM or whatever.
    # In XTTS, that model loads into CUDA/System Ram, so when we swap models, we want to unload the current model
    # free up the memory and then load in the new model to VRAM/CUDA. On the flip side of that, Piper doesnt
    # doesnt load into memory, so we just need to put a fake function here that doesnt really do anything
    # other than set "self.is_tts_model_loaded = False", which would be set back to true by the model loader.
    # So look at the Piper model_engine.py if you DONT need to unload models.
    async def unload_model(self):
        self.is_tts_model_loaded = False
        if not self.current_model_loaded == None:
            print(f"[{self.branding}ENG] \033[94mUnloading model \033[0m") if self.debug_tts else None
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    ###################################################################################################################################
    ###################################################################################################################################
    # CHANGE ME # Model changing. Unload out old model and load in a new one # XTTS is very unusal as it has 2x model loading methods #
    ###################################################################################################################################
    ###################################################################################################################################
    # This function is your central model loading/unloading handler that deals with the above functions as necesary, to call loading, unloading,
    # swappng DeepSpeed, Low vram etc. This function gets called with a "engine name - model name" type call. In XTTS, because there are 2x
    # model loader types, (XTTS and APILocal), we take tts_method and split the "engine name - model name" into a loader type and the model
    # that it needs to load in and then we call the correct loader function. Whereas in Piper, which doesnt load models into memory at all,
    # we just have a fake function that doesnt really do anything. We always check to see if the model name has "No Models Available" in the
    # name thats sent over, just to catch any potential errors. We display the start load time and end load time. Thats about it.
    async def handle_tts_method_change(self, tts_method):
        generate_start_time = time.time() # Record the start time of loading the model
        if "No Models Available" in self.available_models:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No models for this TTS engine were found to load. Please download a model.")
            return False
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        await self.unload_model()
        if tts_method.startswith("llasa"):
            model_name = tts_method.split(" - ")[1]
            print(f"[{self.branding}ENG]\033[94m Model/Engine :\033[93m {model_name}\033[94m loading into\033[93m", self.device,"\033[0m")
            self.model = await self.api_manual_load_model(model_name)
            self.current_model_loaded = f"llasa - {model_name}"
        else:
            self.current_model_loaded = None
            return False

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        generate_end_time = time.time() # Create an end timer for calculating load times
        generate_elapsed_time = generate_end_time - generate_start_time  # Calculate start time minus end time
        print(f"[{self.branding}ENG] \033[94mLoad time :\033[93m {generate_elapsed_time:.2f} seconds.\033[0m") # Print out the result of the load time
        return True #You need to return a True or False statement, based on the outcome of loading your model in

    async def preprocess_ref_audio_text(self, ref_audio_orig, ref_text, clip_short=True):
        print(f"[{self.branding}ENG] Converting audio...") if self.debug_tts else None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            waveform, sample_rate = torchaudio.load(ref_audio_orig)

            # Check if the audio is stereo (i.e., has more than one channel)
            if waveform.size(0) > 1:
                # Convert stereo to mono by averaging the channels
                waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
            else:
                # If already mono, just use the original waveform
                waveform_mono = waveform

            waveform_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform_mono)

            torchaudio.save(f.name, waveform_16k, 16000)
            ref_audio_orig = f.name

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

    ##########################################################################################################################################
    ##########################################################################################################################################
    # CHANGE ME # Model changing. Unload out old model and load in a new one # XTTS is very unusal as it has 2x model TTS generation methods #
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
    # Piper TTS, which uses command line based calls and therefore has different ones for Windows and Linux/Mac, has an example of doing this
    # within its model_engine.py file.
    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming):
        if voice == "No Voices Found":
            print(f"[{self.branding}ENG] \033[91mError\033[0m: No voices found to generate TTS.")
            raise HTTPException(status_code=400, detail="No voices found to generate TTS.")
        if not self.is_tts_model_loaded:
            print(f"[{self.branding}ENG] \033[91mError\033[0m: You currently have no TTS model loaded.")
            raise HTTPException(status_code=400, detail="You currently have no TTS model loaded.")
        self.tts_generating_lock = True # Lock the process to say its currently in use
        generate_start_time = time.time()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        try:
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
            ref_text = None
            if ref_text_path.exists():
                with open(ref_text_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()

            # Process reference audio and text
            ref_audio, processed_ref_text = await self.preprocess_ref_audio_text(
                str(ref_audio_path), ref_text
            )

            # sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=2048, stop=['<|SPEECH_GENERATION_END|>'], stop_token_ids=[128261])
            llm, codec_model = self.model
            audio_out = llasa.text_to_speech(
                llm,
                self.tokenizer,
                codec_model,
                ref_audio,
                text,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                device=self.device,
                prompt_text=processed_ref_text
            )

            sf.write(output_file, audio_out, 16000)

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
            self.tts_generating_lock = False
